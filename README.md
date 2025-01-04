# Customer Churn Prediction Project

This project demonstrates how to preprocess a customer churn dataset, train machine learning models to predict customer churn, and generate a comprehensive report that includes both textual analysis and visualizations.

## Requirements

- Python 3.x
- Poetry (for dependency management)

### Installing Dependencies

1. **Install Poetry** (if you haven't already):
   Follow the instructions from [Poetry's official website](https://python-poetry.org/docs/#installation).

2. **Install the dependencies**:
   After installing Poetry, run the following command in the project directory to install the required dependencies:

   ```bash
   poetry install
   ```

## Usage

1. Run the script using Poetry
```bash
poetry run python main.py
```
This will:
- Load and preprocess the dataset.
- Visualize feature distributions (e.g., histograms of features).
- Train multiple machine learning models (e.g., Random Forest, Logistic Regression, Decision Tree, SVM).
- Evaluate the models using metrics like accuracy and cross-validation.
- Perform hyperparameter tuning on the Random Forest model and save the best model.
- Generate a PDF report that includes the data summary, EDA, model evaluation, and conclusion sections, along with any visualizations (e.g., histograms).

Check [report.pdf](./report.pdf) for an example report.
