import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

#to know where to store the train ,test,raw data we need input locations
#ataclass no need for init constructor if only variables are used

@dataclass
class DataIngestionConfig:
    train_data_path: str= os.path.join('artifacts','train.csv')
    test_data_path : str =os.path.join('artifacts','test.csv')
    raw_data_path : str =os.path.join('artifacts','raw_data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        #to read data from sources
        #logging.info prints like comments 

        logging.info("Entered Data ingestion")

        try:
            df=pd.read_csv('notebook/data/stud.csv')
            logging.info("Data is read as dataframe")

            #to create the artifacts folder and files
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            #if already exists path just add dont recreate like articafts folder once created keep it

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion is completed")

            return (self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path)
        
        except Exception as e:
            raise CustomException(e,sys)


#to check 
if __name__=="__main__":
    obj= DataIngestion()
    obj.initiate_data_ingestion()