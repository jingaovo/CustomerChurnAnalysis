import pandas as pd
import re
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from Constants import DataFields as Fields

BINARY_FEATURES = [Fields.SENIOR_CITIZEN]
TARGET_COLUMN = [Fields.CHURN]


class DataProcessor:

    # -----------------------------------------------------------------------------------------------------------------#
    def __init__(self):
        self.customer_data = pd.read_csv('Customer-Churn.csv')
        self.process_missing_data()
        self.encoding_categorical_data()
        self.split_data()
        self.scale_features()

    # -----------------------------------------------------------------------------------------------------------------#
    def process_missing_data(self):
        missing_summary = self.customer_data.isnull().sum()
        missing_percentage = (missing_summary / len(self.customer_data)) * 100
        threshold = 30
        columns_to_drop = missing_percentage[missing_percentage > threshold].index
        self.customer_data = self.customer_data.drop(columns=columns_to_drop)
        self.customer_data[Fields.TOTAL_CHARGES] = pd.to_numeric(self.customer_data[Fields.TOTAL_CHARGES],
                                                                 errors='coerce')
        numerical_features = self.customer_data.select_dtypes(include=['float64', 'int64']).columns
        numerical_features = numerical_features.drop(BINARY_FEATURES)
        numerical_imputer = SimpleImputer(strategy='mean')
        self.customer_data[numerical_features] = numerical_imputer.fit_transform(self.customer_data[numerical_features])
        self.customer_data.dropna()

    # -----------------------------------------------------------------------------------------------------------------#
    def encoding_categorical_data(self):
        categorical_features = list(self.customer_data.select_dtypes(include=['object']).columns) + BINARY_FEATURES
        categorical_features = [col for col in categorical_features if col != TARGET_COLUMN[0]]
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_features = ohe.fit_transform(self.customer_data[categorical_features])
        encoded_df = pd.DataFrame(encoded_features, columns=ohe.get_feature_names_out(categorical_features),
                                  index=self.customer_data.index)
        self.customer_data = pd.concat([self.customer_data.drop(columns=categorical_features), encoded_df], axis=1)
        target_column = self.customer_data.pop(TARGET_COLUMN[0])
        self.customer_data[TARGET_COLUMN[0]] = target_column

    # -----------------------------------------------------------------------------------------------------------------#
    def split_data(self):
        X = self.customer_data.drop(columns=TARGET_COLUMN)
        y = self.customer_data[TARGET_COLUMN[0]]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # -----------------------------------------------------------------------------------------------------------------#
    def scale_features(self):
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    # -----------------------------------------------------------------------------------------------------------------#
    def convert_to_constant_format(column_name):
        return re.sub(r'([a-z])([A-Z])', r'\1_\2', column_name).upper()

    # -----------------------------------------------------------------------------------------------------------------#
    def get_customer_data(self):
        return self.customer_data


DataProcessor()
