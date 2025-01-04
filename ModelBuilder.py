from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib


class ModelBuilder:
    # -----------------------------------------------------------------------------------------------------------------#
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = {}

    # -----------------------------------------------------------------------------------------------------------------#
    def add_model(self, name, model):
        self.models[name] = model

    # -----------------------------------------------------------------------------------------------------------------#
    def evaluate_model(self, model):
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))
        print("Accuracy Score:", accuracy_score(self.y_test, y_pred))
        
        # Cross-validation scores
        cross_val_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
        print(f"Cross-validation scores: {cross_val_scores}")
        print(f"Mean Cross-validation score: {cross_val_scores.mean()}")

    # -----------------------------------------------------------------------------------------------------------------#
    def compare_model(self, name):
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found.")

        model = self.models[name]
        self.evaluate_model(model)

    # -----------------------------------------------------------------------------------------------------------------#
    def tune_model(self, model, param_grid):
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_}")
        return grid_search.best_estimator_

    # -----------------------------------------------------------------------------------------------------------------#
    def save_model(self, model, filename):
        joblib.dump(model, filename)
        print(f"Model saved to {filename}")

    # -----------------------------------------------------------------------------------------------------------------#
    def load_model(self, filename):
        model = joblib.load(filename)
        print(f"Model loaded from {filename}")
        return model
