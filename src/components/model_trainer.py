import os
import sys
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor,GradientBoostingRegressor

from src.utils import save_object,evaluate_model
from dataclasses import dataclass


@dataclass
class Modeltrainer_config:
    trained_model_path = os.path.join("artifacts",'model.pkl')

class Modeltrainer:
    def __init__(self):
        self.model_trainer_config=Modeltrainer_config

    def initiate_modeltrainer(self,train_arr,test_arr):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models={
                "Random Forest": RandomForestRegressor(),
                "XGB Regressor" : XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(),
                "Linear Regressor": LinearRegression(),
                "Gradient Boosting Regressor" : GradientBoostingRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Decision Tree Regressor":DecisionTreeRegressor(),
                "K-Neighbours Regressor": KNeighborsRegressor()
            }
            params={
                "Random Forest": {'n_estimators': [8, 16, 32, 64, 128, 256]},

                "XGB Regressor" : {
                    'learning_rate':[.1,.01,0.05,0.001],
                    'n_estimators':[8,16,24,32,40,64,128,256]
                },
                "CatBoost Regressor": {
                    'depth': [6,8,10],
                    'learning_rate':[0.01,0.05,0.1],
                    'iterations':[30,40,50,100]
                },
                "Linear Regressor": {},
            
                "Gradient Boosting Regressor":{
                    'learning_rate':[.1,.01,0.05,0.001],
                    'subsample':[0.6,0.65,0.7,0.75,0.8,0.85],
                    'n_estimators':[8,16,24,32,40,64,128,256]
                },
            
               "AdaBoost Regressor": {
                   'learning_rate':[.1,.01,0.05,0.001],
                    'n_estimators':[8,16,24,32,40,64,128,256]
               },
               "Decision Tree Regressor": {'criterion':['squared_error','friedman_mse','absolute_error']},
               "K-Neighbours Regressor": {'n_neighbors':[5,6,7,9,11]}
                
            }

            model_report :dict =evaluate_model(X_train=X_train,y_train=y_train,
                                                X_test=X_test,y_test=y_test, models=models,param=params)

            #To get best model score
            best_model_score = max(sorted(model_report.values()))

            #To get best model name
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)] #Gives indices of highest r2 score 

            best_model = models[best_model_name]

            #random threshold for r2 score
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info("Best model found on both train and test")

            save_object(self.model_trainer_config.trained_model_path,
                        obj = best_model)
            
            predicted =best_model.predict(X_test)
            r2_val = r2_score(y_test,predicted)

            return r2_val





        except Exception as e:
            raise CustomException(e,sys)
