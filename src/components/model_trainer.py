import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.pipeline.exception import CustomException
from src.pipeline.logger import logging
from src.pipeline.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

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
                "Decision Tree": { 'criterion': ['squared_error', 'friedman_mse'] },
                "Random Forest":{ 'n_estimators': [64, 128], 'max_features': ['sqrt', 'log2'] },
                "Gradient Boosting":{ 'learning_rate':[.1,.05], 'n_estimators': [64, 128] },
                "Linear Regression":{},
                "XGBRegressor":{ 'learning_rate':[.1,.05], 'n_estimators': [64, 128] },
                "CatBoosting Regressor":{ 'depth': [6, 8], 'iterations': [100, 200] },
                "AdaBoost Regressor":{ 'learning_rate':[.1,.05], 'n_estimators': [32, 64] }
            }

            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test,
                                                y_test=y_test, models=models, params=params)
            
            # ⭐ NEW: Detailed logging to see all model scores
            logging.info("--- FULL MODEL PERFORMANCE REPORT ---")
            logging.info(model_report)
            logging.info("------------------------------------")

            sorted_models = sorted(model_report.items(), key=lambda item: item[1], reverse=True)
            
            best_single_model_name, best_single_model_score = sorted_models[0]
            logging.info(f"Best performing single model is '{best_single_model_name}' with R2 score: {best_single_model_score:.4f}")
            
            if best_single_model_score < 0.6:
                raise CustomException("No single model met the performance threshold.")
            
            top_models = sorted_models[:3]
            estimators = [(name, models[name]) for name, score in top_models]
            
            voting_regressor = VotingRegressor(estimators=estimators, n_jobs=-1)
            
            logging.info("Training the Voting Regressor Ensemble...")
            voting_regressor.fit(X_train, y_train)
            
            y_pred_ensemble = voting_regressor.predict(X_test)
            ensemble_score = r2_score(y_test, y_pred_ensemble)
            logging.info(f"Voting Regressor Ensemble R2 Score: {ensemble_score:.4f}")

            final_model = voting_regressor
            final_score = ensemble_score

            if best_single_model_score > ensemble_score:
                logging.warning("A single model outperformed the ensemble. Selecting the single best model.")
                final_model = models[best_single_model_name]
                final_score = best_single_model_score

            # ⭐ NEW: Crucial check to see what is being saved
            logging.info(f"--- Preparing to save final model ---")
            logging.info(f"Final model chosen is: {final_model.__class__.__name__}")
            logging.info(f"Final score to be reported is: {final_score:.4f}")
            logging.info("------------------------------------")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=final_model
            )
            
            logging.info(f"Saved final model to {self.model_trainer_config.trained_model_file_path}")
            
            return final_score

        except Exception as e:
            raise CustomException(e, sys)