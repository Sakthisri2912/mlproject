import os
import sys
from src.pipeline.exception import CustomException
from src.pipeline.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Imports needed to run the main block
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    artifacts_dir: str = os.path.join('artifacts')
    raw_data_path: str = os.path.join(artifacts_dir, "data.csv")
    train_data_path: str = os.path.join(artifacts_dir, "train.csv")
    test_data_path: str = os.path.join(artifacts_dir, "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion process")
        try:
            source_data_path = 'notebook/data/stud.csv'
            df = pd.read_csv(source_data_path)
            logging.info('Successfully read dataset as dataframe')

            os.makedirs(self.ingestion_config.artifacts_dir, exist_ok=True)
            logging.info(f"Saving raw data to {self.ingestion_config.raw_data_path}")
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Initiating train-test split")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Data ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except FileNotFoundError:
            logging.error(f"Source file not found at {source_data_path}")
            raise CustomException(f"Source file not found at {source_data_path}", sys)
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    r2_score = modeltrainer.initiate_model_trainer(train_arr, test_arr)
    print(f"Final Best model R2 score on test data: {r2_score:.4f}")