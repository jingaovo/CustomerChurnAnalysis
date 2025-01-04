from DataProcessor import DataProcessor
from ModelBuilder import ModelBuilder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from GenerateReport import ReportGenerator


data_processor = DataProcessor('Customer-Churn.csv')
data_processor.process_missing_data()
data_processor.encoding_categorical_data()
data_processor.split_data()
data_processor.scale_features()
data_processor.plot_feature_distribution()

model_builder = ModelBuilder(data_processor.X_train, data_processor.X_test, data_processor.y_train, data_processor.y_test)
model_builder.add_model("Random Forest", RandomForestClassifier())
model_builder.add_model("Logistic Regression", LogisticRegression())
model_builder.add_model("Decision Tree", DecisionTreeClassifier())
model_builder.add_model("SVM", SVC())
model_builder.fit_all_models()

report_generator = ReportGenerator(data_processor, model_builder)
report_generator.generate_report()
