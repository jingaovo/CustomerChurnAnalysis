import pandas as pd
import re
from sklearn.impute import SimpleImputer
from Constants import DataFields as Fields

NUMERICAL_COLS = [Fields.TENURE, Fields.MONTHLY_CHARGES, Fields.TOTAL_CHARGES]


class DataProcessor:

    # -----------------------------------------------------------------------------------------------------------------#
    def __init__(self):
        self.customer_data = pd.read_csv('Customer-Churn.csv')

    # -----------------------------------------------------------------------------------------------------------------#
    def process_missing_data(self):
        numerical_imputer = SimpleImputer(strategy='mean')
        self.customer_data[NUMERICAL_COLS] = numerical_imputer.fit_transform(self.customer_data[NUMERICAL_COLS])

    # -----------------------------------------------------------------------------------------------------------------#
    def convert_to_constant_format(column_name):
        return re.sub(r'([a-z])([A-Z])', r'\1_\2', column_name).upper()
