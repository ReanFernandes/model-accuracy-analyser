# Model Performance Analysis Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-url.com) This Streamlit application provides an interactive dashboard for visualizing and analyzing model performance results, specifically designed to supplement the findings presented in [Your Paper Title/Link - Optional]. It allows users to explore metrics distributions and learning curves based on experimental data stored in a CSV file.

## Overview

The dashboard helps in understanding model behavior by:

1.  **Analyzing Metrics Distributions:** Comparing various performance metrics (like accuracy, confidence, error types) across different experimental configurations (e.g., model name, prompt type, response format) using box plots.
2.  **Analyzing Learning Curves:** Visualizing how model accuracy (`label_accuracy`) scales with the number of training samples (`num_training_samples`), fitting a power law curve to model the learning process, and calculating related efficiency metrics.

## Features

* **Data Loading:** Load results data from a CSV file either via direct upload or by specifying a file path.
* **Flexible Filtering:** Filter the loaded data based on multiple columns and their values (e.g., `set_type`, `set_number`, `model_name`, `response_type`).
* **Metrics Distribution Analysis:**
    * Select one or more performance metrics for the Y-axis.
    * Group data using up to four categorical columns (X-axis, Color, Facet Column, Facet Row).
    * Visualize distributions using box plots.
    * Sort the X-axis for clearer comparisons.
    * Extensive plot customization options (titles, labels, colors, themes, axis ranges, sample counts).
    * View summary statistics (count, mean, median, std, min, max) for the selected groups.
* **Learning Curve Analysis:**
    * Visualize median `label_accuracy` against `num_training_samples`.
    * Group curves by `model_name` or create **composite groups** by combining multiple parameters (e.g., `model_name` + `prompt_type`).
    * Fit a **power law curve** (`y = a - b*x^-c`) to the median accuracy data for each group.
    * Display fitted curve parameters (`a`, `b`, `c`) and goodness-of-fit (R²).
    * Calculate and display **learning efficiency metrics**:
        * Baseline and maximum accuracy achieved.
        * Total and relative improvement.
        * Sample efficiency (improvement per log10 sample).
        * Plateau point (where learning slows significantly).
    * Visualize the **percentage of maximum gain** achieved at different sample sizes using a heatmap.
    * Customize plot appearance (log scale, error bars, reference lines).
* **Exporting:**
    * Download plots as PNG, HTML, or JSON.
    * Download calculated learning curve metrics and fit parameters as CSV files.
* **Data Exploration:** View a sample of the currently filtered data.

## Expected Data Format

The application is designed to work with a CSV file containing results from model evaluation experiments. While somewhat flexible, it works best with columns similar to the original `new_results_consolidated.csv`, including:

* **Grouping Columns (Categorical/Low Cardinality):**
    * `model_name`: Identifier for the model.
    * `training_status`: e.g., 'trained', 'untrained'.
    * `quantisation_status`: e.g., 'full_model'.
    * `training_dataset`, `num_training_samples`, `num_training_domains`, `randomised_training_samples`: Details about training data.
    * `generation_strategy`: e.g., 'greedy', 'temp_025'.
    * `prompt_type`: e.g., 'few_shot', 'zero_shot'.
    * `explanation_type`: e.g., 'structured', 'unstructured'.
    * `response_type`: e.g., 'fact_first', 'answer_first'.
    * `response_format`: e.g., 'markdown', 'number_list'.
    * `evaluation_dataset`: Identifier for the evaluation set (processed into `set_type` and `set_number`).
* **Metric Columns (Numeric):**
    * `label_accuracy`, `combined_accuracy`: Core accuracy metrics.
    * `misclassification_rate`
    * `malformed_label`, `malformed_domain`: Error counts/rates.
    * `total_predictions`, `correct_predictions`, `correct_label_and_domain`: Raw counts.
    * Columns ending in `_accuracy` (e.g., `Constitutional_Law_accuracy`).
    * Columns ending in `_confidence` (e.g., `Contracts_confidence`).
* **Learning Curve Columns:**
    * `num_training_samples`: Number of samples used for training (X-axis).
    * `label_accuracy`: The primary metric used for learning curves (Y-axis).

*Note: The application attempts some automatic processing (like extracting set type/number from `evaluation_dataset` and converting metric columns to numeric). Ensure your CSV structure aligns with these expectations for best results.*

## Setup and Usage

1.  **Prerequisites:**
    * Python 3.8+
    * pip (Python package installer)

2.  **Clone the Repository (Optional):**
    ```bash
    git clone [https://github.com/ReanFernandes/model-accuracy-analyser.git](https://github.com/ReanFernandes/model-accuracy-analyser.git)
    cd model-accuracy-analyser
    ```
    *(Replace `your-username/your-repo-name.git` with your actual repository URL)*

3.  **Install Dependencies:**
    It's recommended to create a virtual environment:
    ```bash
    # Create virtual environment
    python -m venv venv
    # Activate it (Linux/macOS)
    source venv/bin/activate
    # Or activate it (Windows)
    # venv\Scripts\activate

    # Install required packages
    pip install -r requirements.txt
    ```
  

4.  **Run the App:**
    Navigate to the directory containing the script and run:
    ```bash
    streamlit run accuracy-analyser.py
    ```
    

5.  **Using the App:**
    * Open the URL provided by Streamlit (usually `http://localhost:8501`) in your browser.
    * Use the sidebar to:
        * Load your CSV data (either by uploading or providing the file path if the script is local).
        * Apply filters to select the data subset for analysis.
        * Choose a global color palette.
    * Navigate between the "Metrics Analysis" and "Learning Curve Analysis" tabs.
    * Select metrics, grouping columns, and plot options within each tab.
    * Use the "Customize Plot Appearance" expander to fine-tune visualizations.
    * Download plots and data using the export buttons.

## Dependencies

The main libraries used are:

* Streamlit
* Pandas
* Plotly
* NumPy
* SciPy (for curve fitting)


