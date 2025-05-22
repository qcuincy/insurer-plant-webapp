import os
import sys
import json
import warnings
import traceback
import argparse # For CLI arguments

import pandas as pd
import numpy as np
from scipy.stats import poisson

# --- Global Variables for Loaded Model Components ---
# These will be populated by load_model_components()
model_params_global = None
X_columns_global = []
all_unit_types_global_dropdown = [] # For the dropdown/terminal choices
model_loaded_successfully = False
data_for_input_defaults = {} # For Dash UI input defaults

warnings.filterwarnings('ignore')

# --- Configuration for File Paths ---
MODEL_OUTPUT_DIR = os.path.join(os.getcwd(), "model_outputs")
PARAMS_FILE = os.path.join(MODEL_OUTPUT_DIR, 'model_params.csv')
X_COLUMNS_FILE = os.path.join(MODEL_OUTPUT_DIR, 'model_x_columns.json')
UNIT_TYPES_FILE = os.path.join(MODEL_OUTPUT_DIR, 'model_unit_types.json')
DATA_FILE_PATH_FOR_DEFAULTS = os.path.join(os.getcwd(), "data", 'Combined_Data_filtered.csv')


def load_model_components_and_defaults():
    """Loads model parameters, X columns, unit types, and data for UI defaults."""
    global model_params_global, X_columns_global, all_unit_types_global_dropdown, model_loaded_successfully, data_for_input_defaults

    try:
        # Load Model Parameters
        if os.path.exists(PARAMS_FILE):
            params_df = pd.read_csv(PARAMS_FILE)
            if 'Feature' in params_df.columns and 'Coefficient' in params_df.columns:
                model_params_global = pd.Series(params_df['Coefficient'].values, index=params_df['Feature'])
                print_info("Successfully loaded model parameters.")
            else:
                print_error(f"'Feature' or 'Coefficient' column missing in '{PARAMS_FILE}'.")
        else:
            print_error(f"Model parameters file not found at '{PARAMS_FILE}'.")

        # Load X Columns
        if os.path.exists(X_COLUMNS_FILE):
            with open(X_COLUMNS_FILE, 'r') as f:
                X_columns_global = json.load(f)
            print_info("Successfully loaded X_columns list.")
        else:
            print_error(f"Model X_columns file not found at '{X_COLUMNS_FILE}'.")

        # Load Unit Types for the dropdown/choices
        if os.path.exists(UNIT_TYPES_FILE):
            with open(UNIT_TYPES_FILE, 'r') as f:
                all_unit_types_global_dropdown = json.load(f)
            print_info("Successfully loaded unit types for dropdown/choices.")
        else:
            print_error(f"Model unit types file not found at '{UNIT_TYPES_FILE}'.")
            all_unit_types_global_dropdown = ["(Unit Types File Missing)"]

        if model_params_global is not None and X_columns_global:
            missing_in_params = set(X_columns_global) - set(model_params_global.index)
            if missing_in_params:
                print_warning(f"Columns in X_columns_global are missing from loaded model_params keys: {missing_in_params}. Filling with 0 coefficient.")
                for col in missing_in_params: model_params_global[col] = 0.0
            
            extra_in_params = set(model_params_global.index) - set(X_columns_global)
            if extra_in_params:
                print_warning(f"Loaded model_params contains keys not in X_columns_global: {extra_in_params}. These parameters will be ignored.")
            
            model_params_global = model_params_global.reindex(X_columns_global)
            if model_params_global.isnull().any():
                nan_params = model_params_global[model_params_global.isnull()].index.tolist()
                print_warning(f"Model parameters for features {nan_params} are NaN after reindexing. Filling with 0.")
                model_params_global = model_params_global.fillna(0.0)
            
            model_loaded_successfully = True
            print_info("Model components processed and aligned.")
        else:
            print_error("Core model components (parameters or X_columns) failed to load.")
            if not all_unit_types_global_dropdown: all_unit_types_global_dropdown = ["(No Data)"]

        # Load data for input defaults (Optional, for Dash UI)
        if os.path.exists(DATA_FILE_PATH_FOR_DEFAULTS):
            try:
                temp_df = pd.read_csv(DATA_FILE_PATH_FOR_DEFAULTS, delimiter=';', usecols=['MWLost'])
                temp_df.columns = temp_df.columns.str.strip()
                if 'MWLost' in temp_df.columns and not temp_df['MWLost'].empty:
                    data_for_input_defaults['MWLost_median'] = temp_df['MWLost'].median()
                print_info("Successfully loaded data for input defaults.")
            except Exception as e_data_load:
                print_warning(f"Could not load or process data for input defaults from '{DATA_FILE_PATH_FOR_DEFAULTS}': {e_data_load}")
        else:
            print_warning(f"Data file for input defaults not found at '{DATA_FILE_PATH_FOR_DEFAULTS}'. Using static defaults for some inputs.")

    except Exception as e:
        print_error(f"An error occurred during initial loading: {e}")
        traceback.print_exc()
        if not all_unit_types_global_dropdown: all_unit_types_global_dropdown = ["(Error Loading Unit Types)"]
        model_loaded_successfully = False

def get_core_predictions(unit_type_input, mw_lost_val, minor_count_val, avg_severity_val,
                         model_params, x_cols, unit_type_options_list):
    """Calculates core prediction values."""
    current_input_row_dict = {'Intercept': 1.0, 'MWLost': mw_lost_val, 'minor_count': minor_count_val}
    for ut_col_base in unit_type_options_list:
        if ut_col_base.startswith("("): continue
        formatted_ut_col = f'UnitType_{ut_col_base}'
        current_input_row_dict[formatted_ut_col] = 1.0 if ut_col_base == unit_type_input else 0.0
    
    input_df_ordered = pd.DataFrame([current_input_row_dict]).reindex(columns=x_cols, fill_value=0.0).astype(float)
    input_vector = input_df_ordered.iloc[0].reindex(model_params.index, fill_value=0.0)
    
    log_lambda = (model_params * input_vector).sum()
    lambda_pred = np.exp(log_lambda)

    predictions = {
        "lambda_pred": lambda_pred,
        "prob_zero_events": poisson.pmf(0, lambda_pred),
        "prob_one_event": poisson.pmf(1, lambda_pred),
        "prob_two_events": poisson.pmf(2, lambda_pred),
    }
    predictions["prob_three_plus_events"] = 1 - (predictions["prob_zero_events"] + predictions["prob_one_event"] + predictions["prob_two_events"]) # More direct CDF
    predictions["prob_at_least_one_event"] = 1 - predictions["prob_zero_events"]

    premium_text_parts_list = []
    if avg_severity_val > 0:
        illustrative_pure_premium = lambda_pred * avg_severity_val
        premium_text_parts_list.append(f"Assumed Average Severity: ${avg_severity_val:,.0f}")
        premium_text_parts_list.append(f"Illustrative Pure Premium: ${illustrative_pure_premium:,.2f} per year")
        premium_text_parts_list.append("Note: This is a simplified Pure Premium.")
    else:
        premium_text_parts_list.append("Illustrative premium pricing not calculated (Severity not provided or zero).")
    predictions["premium_text"] = "\n".join(premium_text_parts_list)
    
    return predictions

def display_terminal_output(predictions):
    """Prints formatted predictions to the terminal."""
    print_info("\n--- Prediction Summary ---")
    print(f"Expected 30+ Day Outages (λ): {predictions['lambda_pred']:.4f}")
    print("-" * 30)
    print(f"P(X=0) Outages:  {predictions['prob_zero_events']:.4f} ({predictions['prob_zero_events']*100:.2f}%)")
    print(f"P(X=1) Outage:   {predictions['prob_one_event']:.4f} ({predictions['prob_one_event']*100:.2f}%)")
    print(f"P(X=2) Outages:  {predictions['prob_two_events']:.4f} ({predictions['prob_two_events']*100:.2f}%)")
    print(f"P(X>=3) Outages: {predictions['prob_three_plus_events']:.4f} ({predictions['prob_three_plus_events']*100:.2f}%)")
    print("-" * 30)
    print(f"P(At Least One 30+ Day Outage): {predictions['prob_at_least_one_event']:.4f} ({predictions['prob_at_least_one_event']*100:.2f}%)")
    print("-" * 30)
    print("--- Illustrative Premium Pricing ---")
    print(predictions['premium_text'])
    print("-" * 30)

def run_terminal_mode():
    """Handles user interaction and output for terminal mode."""
    global model_params_global, X_columns_global, all_unit_types_global_dropdown
    print_info("Running in Terminal Mode...")

    if not all_unit_types_global_dropdown or all_unit_types_global_dropdown[0].startswith("("):
        print_error("Unit types not available. Cannot proceed with terminal input.")
        return

    print("\nAvailable Plant Technologies:")
    for i, tech in enumerate(all_unit_types_global_dropdown):
        print(f"  {i+1}. {tech}")
    
    while True:
        try:
            choice_idx = int(input(f"Select Plant Technology (1-{len(all_unit_types_global_dropdown)}): ")) - 1
            if 0 <= choice_idx < len(all_unit_types_global_dropdown):
                unit_type = all_unit_types_global_dropdown[choice_idx]
                break
            else:
                print_error("Invalid choice. Please select a number from the list.")
        except ValueError:
            print_error("Invalid input. Please enter a number.")

    while True:
        try:
            mw_lost_str = input("Enter Plant Size (MW) (e.g., 100): ")
            mw_lost = float(mw_lost_str)
            if mw_lost >= 0: break
            else: print_error("Plant size must be non-negative.")
        except ValueError:
            print_error("Invalid input. Please enter a number for Plant Size.")
    
    while True:
        try:
            minor_count_str = input("Enter Minor Events (1-30 days) (e.g., 1): ")
            minor_count = int(minor_count_str)
            if minor_count >= 0: break
            else: print_error("Minor events count must be non-negative.")
        except ValueError:
            print_error("Invalid input. Please enter an integer for Minor Events.")

    while True:
        try:
            avg_severity_str = input("Enter Assumed Avg Severity ($) (e.g., 500000, press Enter to skip): ")
            if not avg_severity_str: # User pressed Enter
                avg_severity = 0.0
                break
            avg_severity = float(avg_severity_str)
            if avg_severity >= 0: break
            else: print_error("Severity must be non-negative.")
        except ValueError:
            print_error("Invalid input. Please enter a number for Severity or press Enter to skip.")

    predictions = get_core_predictions(unit_type, mw_lost, minor_count, avg_severity,
                                       model_params_global, X_columns_global, all_unit_types_global_dropdown)
    display_terminal_output(predictions)

# --- Dash App specific functions and imports ---
# These are defined here to be conditionally imported/used
dash_app_instance = None
dcc_global = None
html_global = None
Input_global = None
Output_global = None
State_global = None
dbc_global = None
go_global = None

def _initialize_dash_components():
    """Dynamically imports Dash components."""
    global dash_app_instance, dcc_global, html_global, Input_global, Output_global, State_global, dbc_global, go_global
    try:
        import dash
        from dash import dcc, html, Input, Output, State
        import dash_bootstrap_components as dbc
        import plotly.graph_objects as go

        dash_app_instance = dash
        dcc_global = dcc
        html_global = html
        Input_global = Input
        Output_global = Output
        State_global = State
        dbc_global = dbc
        go_global = go
        return True
    except ImportError:
        print_error("Dash libraries not installed. Please install them to run the web app mode:")
        print_info("  pip install dash dash-bootstrap-components plotly pandas")
        return False

def create_dash_app_layout():
    """Creates the Dash app layout."""
    global all_unit_types_global_dropdown, data_for_input_defaults, html_global, dbc_global, dcc_global
    return dbc_global.Container([
        dbc_global.Row(dbc_global.Col(html_global.H1("Power Plant Outage Predictor", className="text-center my-4"), width=12)),
        dbc_global.Card(
            dbc_global.CardBody([
                dbc_global.Row([
                    dbc_global.Col([
                        dbc_global.Label("Plant Technology:", html_for="unitType"),
                        dcc_global.Dropdown(
                            id='unitType',
                            options=[{'label': i, 'value': i} for i in all_unit_types_global_dropdown],
                            value=all_unit_types_global_dropdown[0] if all_unit_types_global_dropdown and not all_unit_types_global_dropdown[0].startswith("(") else None,
                            clearable=False)], md=3, className="mb-3 mb-md-0"),
                    dbc_global.Col([
                        dbc_global.Label("Plant Size (MW):", html_for="mwLost"),
                        dbc_global.Input(id='mwLost', type='number', value=data_for_input_defaults.get('MWLost_median', 100.0), min=0)], md=3, className="mb-3 mb-md-0"),
                    dbc_global.Col([
                        dbc_global.Label("Minor Events (1-30 days):", html_for="minorCount"),
                        dbc_global.Input(id='minorCount', type='number', value=1, min=0, step=1)], md=3, className="mb-3 mb-md-0"),
                    dbc_global.Col([
                        dbc_global.Label("Assumed Avg Severity ($):", html_for="avgSeverity"),
                        dbc_global.Input(id='avgSeverity', type='number', value=500000, min=0, step=10000)], md=3, className="mb-3 mb-md-0"),
                ], className="align-items-end"),
            ]), className="mb-4 shadow", id="input-card"),
        dbc_global.Row([
            dbc_global.Col([
                dbc_global.Card([
                    dbc_global.CardHeader("Prediction Summary"),
                    dbc_global.CardBody([
                        html_global.Div(id='textResults', children=[
                            html_global.P([html_global.Strong("Expected 30+ Day Outages (λ): "), html_global.Span(id="lambdaPred", className="font-weight-bold")]),
                            html_global.Hr(),
                            html_global.P([html_global.Strong("P(X=0) Outages: "), html_global.Span(id="probZero"), " (", html_global.Span(id="probZeroPercent"), "%)"]),
                            html_global.P([html_global.Strong("P(X=1) Outage: "), html_global.Span(id="probOne"), " (", html_global.Span(id="probOnePercent"), "%)"]),
                            html_global.P([html_global.Strong("P(X=2) Outages: "), html_global.Span(id="probTwo"), " (", html_global.Span(id="probTwoPercent"), "%)"]),
                            html_global.P([html_global.Strong("P(X>=3) Outages: "), html_global.Span(id="probThreePlus"), " (", html_global.Span(id="probThreePlusPercent"), "%)"]),
                            html_global.Hr(),
                            html_global.P([html_global.Strong("P(At Least One 30+ Day Outage): "), html_global.Span(id="probAtLeastOne"), " (", html_global.Span(id="probAtLeastOnePercent"), "%)"]),
                            html_global.Hr(),
                            html_global.Div(id="premiumPricing")])])], className="mb-4 shadow h-100")], md=6),
            dbc_global.Col([
                dbc_global.Card([
                    dbc_global.CardHeader("Expected Outages (λ) Gauge"),
                    dbc_global.CardBody([dcc_global.Graph(id='lambdaGaugeChart', config={'displayModeBar': False})], style={"minHeight": "300px"})], className="mb-4 shadow h-100")], md=6)]),
        dbc_global.Row([
            dbc_global.Col(dbc_global.Card([dbc_global.CardHeader("Probability Distribution of 30+ Day Outages"), dbc_global.CardBody([dcc_global.Graph(id='probDistChart')])], className="shadow"), md=6, className="mb-4"),
            dbc_global.Col(dbc_global.Card([dbc_global.CardHeader("Probability Breakdown (Pie Chart)"), dbc_global.CardBody([dcc_global.Graph(id='probPieChart')])], className="shadow"), md=6, className="mb-4")]),
        dbc_global.Row([dbc_global.Col(dbc_global.Card([dbc_global.CardHeader("Sensitivity to Minor Events"), dbc_global.CardBody([dcc_global.Graph(id='sensitivityChart')])], className="shadow"), width=12, className="mb-4")])
    ], fluid=True, className="p-4 bg-light")

def register_dash_callbacks(app):
    """Registers callbacks for the Dash app."""
    global model_params_global, X_columns_global, all_unit_types_global_dropdown, model_loaded_successfully, html_global, go_global, dash_app_instance

    @app.callback(
        [Output_global('lambdaPred', 'children'),
         Output_global('probZero', 'children'), Output_global('probZeroPercent', 'children'),
         Output_global('probOne', 'children'), Output_global('probOnePercent', 'children'),
         Output_global('probTwo', 'children'), Output_global('probTwoPercent', 'children'),
         Output_global('probThreePlus', 'children'), Output_global('probThreePlusPercent', 'children'),
         Output_global('probAtLeastOne', 'children'), Output_global('probAtLeastOnePercent', 'children'),
         Output_global('premiumPricing', 'children'),
         Output_global('probDistChart', 'figure'),
         Output_global('sensitivityChart', 'figure'),
         Output_global('lambdaGaugeChart', 'figure'),
         Output_global('probPieChart', 'figure')],
        [Input_global('unitType', 'value'),
         Input_global('mwLost', 'value'),
         Input_global('minorCount', 'value'),
         Input_global('avgSeverity', 'value')]
    )
    def update_predictions_and_charts_for_dash(unit_type, mw_lost, minor_count_in, avg_severity):
        if not model_loaded_successfully or model_params_global is None or not X_columns_global:
            no_model_text = "Model components not loaded."
            empty_fig = go_global.Figure().update_layout(title="Model Not Loaded")
            return ([no_model_text] * 12 + [[no_model_text]] + [empty_fig]*4) # Adjusted count

        if unit_type is None or mw_lost is None or minor_count_in is None or avg_severity is None:
            return dash_app_instance.no_update

        try:
            mw_lost_val = float(mw_lost)
            minor_count_val = int(minor_count_in)
            avg_severity_val = float(avg_severity)

            core_preds = get_core_predictions(unit_type, mw_lost_val, minor_count_val, avg_severity_val,
                                              model_params_global, X_columns_global, all_unit_types_global_dropdown)
            
            lambda_pred = core_preds["lambda_pred"] # For sensitivity chart

            # Convert premium text for HTML
            premium_html_parts = [html_global.P(part) for part in core_preds["premium_text"].split("\n")]


            # Create Figures for Dash
            labels_bar = ['P(X=0)', 'P(X=1)', 'P(X=2)', 'P(X>=3)']
            probabilities_bar = [core_preds['prob_zero_events'], core_preds['prob_one_event'], core_preds['prob_two_events'], core_preds['prob_three_plus_events']]
            fig_prob_dist = go_global.Figure(data=[go_global.Bar(x=labels_bar,y=probabilities_bar,text=[f'{p:.3f}' for p in probabilities_bar],textposition='auto',marker_color=['#1f77b4','#ff7f0e','#2ca02c','#d62728'])])
            fig_prob_dist.update_layout(yaxis_title='Probability',yaxis_range=[0, max(probabilities_bar)*1.2 if probabilities_bar and max(probabilities_bar)>0 else 1],margin=dict(t=20,b=20,l=30,r=20),height=350,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(color="#505050"))

            minor_counts_range = np.arange(max(0, minor_count_val - 5), minor_count_val + 6)
            probs_at_least_one_sensitivity = []
            for mc_sens in minor_counts_range:
                sens_preds = get_core_predictions(unit_type, mw_lost_val, float(mc_sens), avg_severity_val,
                                                  model_params_global, X_columns_global, all_unit_types_global_dropdown)
                probs_at_least_one_sensitivity.append(sens_preds['prob_at_least_one_event'])

            fig_sensitivity = go_global.Figure()
            fig_sensitivity.add_trace(go_global.Scatter(x=minor_counts_range,y=probs_at_least_one_sensitivity,mode='lines+markers',name='P(X>=1) Sensitivity'))
            fig_sensitivity.add_trace(go_global.Scatter(x=[minor_count_val],y=[core_preds['prob_at_least_one_event']],mode='markers',marker=dict(color='red',size=12,symbol='star'),name=f'Current ({minor_count_val} events)'))
            fig_sensitivity.update_layout(xaxis_title='Num of Minor Events',yaxis_title='P(At Least One Major Outage)',showlegend=True,margin=dict(t=20,b=20,l=30,r=20),height=350,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(color="#505050"))

            max_lambda_gauge = max(0.5, core_preds['lambda_pred'] * 1.5, 1.0)
            fig_gauge = go_global.Figure(go_global.Indicator(mode="gauge+number",value=core_preds['lambda_pred'],title={'text':"λ (Expected Outages/Year)",'font':{'size':16}},gauge={'axis':{'range':[0,max_lambda_gauge],'tickwidth':1,'tickcolor':"darkblue"},'bar':{'color':"darkblue"},'bgcolor':"white",'borderwidth':2,'bordercolor':"gray",'steps':[{'range':[0,max_lambda_gauge*0.33],'color':'lightgreen'},{'range':[max_lambda_gauge*0.33,max_lambda_gauge*0.66],'color':'yellow'},{'range':[max_lambda_gauge*0.66,max_lambda_gauge],'color':'red'}],'threshold':{'line':{'color':"black",'width':4},'thickness':0.75,'value':core_preds['lambda_pred'] if core_preds['lambda_pred']<=max_lambda_gauge else max_lambda_gauge}}))
            fig_gauge.update_layout(margin=dict(t=40,b=10,l=30,r=30),height=280,paper_bgcolor='rgba(0,0,0,0)')

            initial_labels_pie = ['P(X=0)','P(X=1)','P(X=2)','P(X>=3)']
            initial_values_pie = [core_preds['prob_zero_events'],core_preds['prob_one_event'],core_preds['prob_two_events'],core_preds['prob_three_plus_events']]
            display_labels_pie, display_values_pie, display_pull_values = [], [], []
            plot_threshold, pull_threshold_small, pull_amount = 0.01, 0.15, 0.1
            sum_initial_values = sum(initial_values_pie) if sum(initial_values_pie) > 0 else 1
            for i in range(len(initial_values_pie)):
                if initial_values_pie[i] >= plot_threshold:
                    display_labels_pie.append(initial_labels_pie[i])
                    display_values_pie.append(initial_values_pie[i])
                    current_pull = pull_amount if (initial_values_pie[i] / sum_initial_values) < pull_threshold_small else 0
                    display_pull_values.append(current_pull)
            
            if not display_labels_pie:
                fig_pie = go_global.Figure().update_layout(title_text="All probabilities < 1%", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            else:
                fig_pie=go_global.Figure(data=[go_global.Pie(labels=display_labels_pie,values=display_values_pie,hole=.3,pull=display_pull_values)])
                fig_pie.update_traces(textinfo='percent+label',marker=dict(colors=['#1f77b4','#ff7f0e','#2ca02c','#d62728']))
                fig_pie.update_layout(margin=dict(t=20,b=20,l=20,r=20),height=350,showlegend=True,paper_bgcolor='rgba(0,0,0,0)',font=dict(color="#505050"),legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))

            return (f"{core_preds['lambda_pred']:.4f}",
                    f"{core_preds['prob_zero_events']:.4f}", f"{core_preds['prob_zero_events']*100:.2f}",
                    f"{core_preds['prob_one_event']:.4f}", f"{core_preds['prob_one_event']*100:.2f}",
                    f"{core_preds['prob_two_events']:.4f}", f"{core_preds['prob_two_events']*100:.2f}",
                    f"{core_preds['prob_three_plus_events']:.4f}", f"{core_preds['prob_three_plus_events']*100:.2f}",
                    f"{core_preds['prob_at_least_one_event']:.4f}", f"{core_preds['prob_at_least_one_event']*100:.2f}",
                    premium_html_parts,
                    fig_prob_dist, fig_sensitivity, fig_gauge, fig_pie)

        except Exception as e:
            print_error(f"Error in Dash callback: {e}")
            traceback.print_exc()
            error_text = f"Error: {str(e)}"
            empty_fig_error = go_global.Figure().update_layout(title=error_text)
            return ([error_text] * 12 + [[error_text]] + [empty_fig_error]*4) # Adjusted count

def run_dash_mode():
    """Initializes and runs the Dash web application."""
    global dash_app_instance, server # server is needed for gunicorn
    if _initialize_dash_components():
        app = dash_app_instance.Dash(__name__, external_stylesheets=[dbc_global.themes.LUMEN])
        app.layout = create_dash_app_layout()
        register_dash_callbacks(app)
        server = app.server # Expose server for WSGI
        
        print_info("Attempting to start Dash server on http://127.0.0.1:8050/ ...")
        try:
            app.run(debug=True)
        except SystemExit as se:
            print_info(f"Dash server exited with code: {se.code}")
            if se.code == 1 and not os.environ.get("WERKZEUG_RUN_MAIN"):
                 print_warning("This SystemExit might indicate the port is in use or an internal Dash/Flask error.")
        except Exception as e:
            print_error(f"CRITICAL ERROR: Failed to start Dash server: {e}")
            traceback.print_exc()
    else:
        print_info("Dash mode cannot be started due to missing libraries.")

def print_info(message): print(f"[INFO] {message}")
def print_warning(message): print(f"[WARNING] {message}")
def print_error(message): print(f"[ERROR] {message}")

# --- Main Execution Logic ---
if __name__ == '__main__':
    load_model_components_and_defaults()

    if not model_loaded_successfully:
        print_error("="*80)
        print_error("STOPPING APPLICATION: MODEL COMPONENTS NOT LOADED SUCCESSFULLY.")
        print_error(f"Please ensure model files exist in '{MODEL_OUTPUT_DIR}' and are readable.")
        print_error("Run the model training/export script to generate these files.")
        print_error("="*80)
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description="Power Plant Outage Predictor CLI")
    parser.add_argument(
        "--mode", 
        choices=['terminal', 'dash'], 
        help="Mode to run the application: 'terminal' for CLI, 'dash' for web app."
    )
    args = parser.parse_args()

    chosen_mode = args.mode
    if not chosen_mode:
        while True:
            choice = input("Run in (t)erminal or (d)ash web app mode? [t/d]: ").lower().strip()
            if choice == 't':
                chosen_mode = 'terminal'
                break
            elif choice == 'd':
                chosen_mode = 'dash'
                break
            else:
                print_error("Invalid choice. Please enter 't' or 'd'.")
    
    if chosen_mode == 'terminal':
        run_terminal_mode()
    elif chosen_mode == 'dash':
        run_dash_mode()