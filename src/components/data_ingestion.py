import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass   # use to creatclass variable 

@dataclass   # Decorator  why - because inside a class to define the class variable we use init but by using this we can directly define the data classes 
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def intiate_data_ingestion(self):
        logging.info("Entered the data ingestion or component")
        try:
            df = pd.read_csv('notebook\data/boston_clean.csv')
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index = False,header=True)
            logging.info("Train test split initiated")
            train_set,test_set = train_test_split(df,test_size=0.3,random_state=42)
            

        except:
            pass
