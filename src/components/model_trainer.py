import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64]
                },
                "Gradient Boosting": {
                    'learning_rate': [.1, .01],
                    'n_estimators': [8, 16, 32]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [.1, .01],
                    'n_estimators': [8, 16, 32]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8],
                    'learning_rate': [0.01, 0.05],
                    'iterations': [30, 50]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01],
                    'n_estimators': [8, 16, 32]
                }
            }

            # Evaluate models
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            # ✅ FIX: get trained best model
            best_model_name = max(model_report, key=lambda x: model_report[x]["score"])
            best_model = model_report[best_model_name]["model"]
            best_model_score = model_report[best_model_name]["score"]

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)

            logging.info(f"Best model found: {best_model_name}")

            # ✅ Save TRAINED model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Evaluate
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)