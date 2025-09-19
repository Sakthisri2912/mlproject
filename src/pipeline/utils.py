import os
import sys
import dill
import numpy as np 
import pandas as pd
import math
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from src.pipeline.exception import CustomException
from src.pipeline.logger import logging

def save_object(file_path, obj):
    """Saves a Python object to a file using dill."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    Evaluates machine learning models and returns a report.
    """
    try:
        report = {}
        for model_name, model in models.items():
            logging.info(f"Evaluating model: {model_name}")
            
            param_grid = params.get(model_name, {})

            # Special handling for models with no hyperparameters
            if not param_grid:
                logging.info(f"Fitting {model_name} directly (no parameters to tune).")
                model.fit(X_train, y_train)
            else:
                # Use GridSearchCV for exhaustive search on defined params
                search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
                search.fit(X_train, y_train)
                model.set_params(**search.best_params_)
                logging.info(f"Best params for {model_name}: {search.best_params_}")
                
                # ⭐ FIX: Re-train the model on the full training data with the best parameters
                model.fit(X_train, y_train)
            
            # Make predictions and evaluate
            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[model_name] = test_model_score
            logging.info(f"Model: {model_name}, R2 Score: {test_model_score:.4f}")

        return report
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """Loads a Python object from a file."""
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)