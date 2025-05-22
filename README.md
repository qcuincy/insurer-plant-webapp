# Power Plant Outage Predictor Application

This application predicts the probability and expected number of major power plant outages based on various plant characteristics. It uses a pre-trained Poisson GLM model.

The application can be run in two modes:
1.  **Terminal Mode:** Interactive command-line interface for quick predictions.
2.  **Dash Web App Mode:** A full web-based dashboard with interactive inputs and visualizations.

## Prerequisites

Before you can run this application, you need the following:

1.  **Python 3.7+:** Ensure you have Python installed on your system. You can download it from [python.org](https://www.python.org/).
2.  **Model Files:** The application relies on pre-trained model components. You must first run the `train_and_export_model.py` script (or an equivalent script that generates these files). This script will create a `model_outputs` directory in your project root with the following files:
    * `model_params.csv` (model coefficients)
    * `model_x_columns.json` (list of predictor variable names)
    * `model_unit_types.json` (list of unique plant technology types)
3.  **Data for Input Defaults (Optional but Recommended for Dash UI):**
    * Create a `data` directory in your project root.
    * Place your `Combined_Data_filtered.csv` file inside this `data` directory. This file is used by the Dash application to set sensible default values for some input fields (e.g., median plant size). If this file is not present, static defaults will be used.

## Setup

1.  **Download Application Files:**
    * Ensure you have the `app.py` script (the main application file).
    * Make sure the `model_outputs` directory (from Prerequisites step 2) is in the same main directory as `app.py`.
    * Make sure the `data` directory (from Prerequisites step 3) is in the same main directory as `app.py`.

2.  **Create a Virtual Environment (Recommended):**
    It's good practice to create a virtual environment to manage project dependencies.
    Open your terminal or command prompt in the project's root directory and run:
    ```bash
    python -m venv venv
    ```
    Activate the virtual environment:
    * On Windows:
        ```bash
        venv\Scripts\activate
        ```
    * On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```

3.  **Install Dependencies:**
    Create a `requirements.txt` file in your project's root directory with the following content:
    ```text
    pandas
    numpy
    scipy
    argparse
    # For Dash mode:
    dash
    dash-bootstrap-components
    plotly
    ```
    Then, install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `statsmodels` is used in the training script but not directly by `app.py` if you're only using the pre-calculated parameters.*

## Running the Application

You can run the application from your terminal, in the project's root directory (ensure your virtual environment is activated).

The script `app.py` supports a `--mode` argument to choose how to run:

1.  **Interactive Prompt (Default):**
    If you run the script without any arguments, it will prompt you to choose the mode:
    ```bash
    python app.py
    ```
    It will ask: `Run in (t)erminal or (d)ash web app mode? [t/d]:`
    * Enter `t` for Terminal Mode.
    * Enter `d` for Dash Web App Mode.

2.  **Terminal Mode:**
    To run directly in terminal mode:
    ```bash
    python app.py --mode terminal
    ```
    The application will then guide you through entering the necessary inputs:
    * Plant Technology (select from a list)
    * Plant Size (MW)
    * Number of Minor Events
    * Assumed Average Severity (optional, press Enter to skip)
    The prediction results will be printed directly in your terminal.
<p align="center">
  <img src="screenshots\terminal.png" width="350" title="hover text">
</p>
3.  **Dash Web App Mode:**
    To run as a web application:
    ```bash
    python app.py --mode dash
    ```
    * **Important:** This mode requires Dash and its related libraries (`dash-bootstrap-components`, `plotly`) to be installed. If they are not installed, the script will notify you.
    * If the libraries are present, the Dash application will start. By default, it will be accessible at: `http://127.0.0.1:8050/`
    * Open this URL in your web browser to use the interactive dashboard.
    * To stop the Dash server, go back to your terminal and press `Ctrl+C`.
<p align="center">
  <img src="screenshots\dash_app.png" width="350" title="hover text">
</p>
## File Structure Overview

For the application to run correctly, your main project directory should ideally look like this:


your_project_directory/
├── app.py                     # The main application script
├── model_outputs/             # Contains pre-trained model files
│   ├── model_params.csv
│   ├── model_x_columns.json
│   └── model_unit_types.json
├── data/                      # Contains data for UI defaults
│   └── Combined_Data_filtered.csv
├── requirements.txt           # Python dependencies
└── venv/                      # Virtual environment (optional but recommended)


Make sure the `train_and_export_model.py` script has been run successfully to populate the `model_outputs` directory before attempting to run `app.py`.
