import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config =DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            num=['writing_score','reading_score']
            cat=['gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course']
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            cat_pipeline=Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("onehotencoder",OneHotEncoder(handle_unknown='ignore')),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info("Categorical columns : {cat}")
            logging.info("Numerical Features : {num}")

            preprocessor = ColumnTransformer(
                [
                    ("Numerical pipeline",num_pipeline,num),
                    ("Categorical pipeline",cat_pipeline,cat)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def inititate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data")

            logging.info("Obtaining preprocessing objects")
            preprocessor_obj = self.get_data_transformer_obj()

            target = "math_score"
            num = ['writing_score','reading_score']
            input_train_df = train_df.drop(columns=[target],axis=1) #X_train
            target_train_df = train_df[target]                       #Y_train

            input_test_df = test_df.drop(columns=[target],axis=1)      #X_test
            target_test_df = test_df[target]                            #Y_test

            logging.info("Applying Preprocessing")
            input_train = preprocessor_obj.fit_transform(input_train_df)
            input_test = preprocessor_obj.transform(input_test_df) 

            train_arr = np.c_[
                input_train,np.array(target_train_df)
            ]

            test_arr = np.c_[
                input_test,np.array(target_test_df)
            ]
            logging.info("Saved Preprocessed object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            return (train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path )
        

        except Exception as e:
            raise CustomException(e,sys)
        
    




