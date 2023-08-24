# This file is still in building
# Main module
from load.load_data import DataRetriever
import pandas as pd
import joblib
from train.train_data import OnlineFraudPipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
# from fastapi import FastAPI
# Body,
# Path,
# Query,
# HTTPException,
# status
# import os
from utilities.logging_util import LoggingSetter
from predictor.predict import ModelPredictor

# SETTING THE LOGGER UTILITY
# Using the logger utility for handling log setting and creation
loggerSet = LoggingSetter(__name__)

# All logs will be saved at utilities folder
logger = loggerSet.setting_log('utilities/main.log')

# Constants for the load process
# This version now includes a function in load to dowload the balanced dataset from GDrive
# The file is too big, so to use the whole data set there is a version we use a local path instead of an URL
GDRIVE_URL = "https://drive.google.com/uc?id=1vW9gp6RFLRHrLBOz4PmN3mwLC_j86j-w&export=download"
FILE_PATH = "c:/Users/oscar.betanzos/Documents/Dataset/onlinefraud.csv"
DATASETS_DIR = './data/'
RETRIEVED_DATA = 'retrieved_data.csv'

# Constants/Parameters for the train process
SEED_MODEL = 404
SELECTED_FEATURES = ['amount', 'oldbalanceOrg', 'newbalanceOrig']
CATEGORICAL_VARS = ['type']
TARGET = 'isFraud'
TEST_SPLIT = 0.25

# Constants for the models export process
TRAINED_MODEL_DIR = './models/'
PIPELINE_NAME = 'logistic_regression'
PIPELINE_SAVE_FILE = f'{PIPELINE_NAME}_output.pkl'


# Retrieve data
logger.debug("Calling DataRetriever class")
data_retriever = DataRetriever(FILE_PATH, DATASETS_DIR)
# result = data_retriever.retrieve_data()
result = data_retriever.retrieve_data_Gdrive(GDRIVE_URL)
logger.debug(f"Result of DataRetriever.data_retreiver:  {result}")
# print(result)

# Read data
data = pd.read_csv(DATASETS_DIR + RETRIEVED_DATA)

# Preview of the data at log
logger.debug(f"Data head: \n {data.head()}")

# Instantiate the Pipeline class
logger.debug("Calling OnlineFraudPipeline Class")
onlinefraud_data_pipeline = OnlineFraudPipeline(seed_model=SEED_MODEL,
                                                categorical_vars=CATEGORICAL_VARS,
                                                selected_features=SELECTED_FEATURES)


# Split data into sets for training and testing
X_train, X_test, y_train, y_test = train_test_split(data.drop(TARGET, axis=1),
                                                    data[TARGET],
                                                    test_size=TEST_SPLIT,
                                                    random_state=SEED_MODEL)


logistic_regression_model = onlinefraud_data_pipeline.fit_logistic_regression(X_train, y_train)

X_test = onlinefraud_data_pipeline.PIPELINE.fit_transform(X_test)
y_pred = logistic_regression_model.predict(X_test)

# Making predictions and measuring performance-
class_pred = logistic_regression_model.predict(X_test)
proba_pred = logistic_regression_model.predict_proba(X_test)[:, 1]
logger.debug(f'test roc-auc : {roc_auc_score(y_test, proba_pred)}')
logger.debug(f'test accuracy: {accuracy_score(y_test, class_pred)}')

# Displaying confusion matrix
conf_mx = confusion_matrix(y_test, y_pred, labels=logistic_regression_model.classes_)
logger.debug(f"Confusion matrix: \n {conf_mx}")

# Save the model using joblib
save_path = TRAINED_MODEL_DIR + PIPELINE_SAVE_FILE
joblib.dump(logistic_regression_model, save_path)
logger.debug(f"Model saved in {save_path}")

# API predict test
data_aux = {'type': ['CASH_OUT'],
            'amount': [1000],
            'oldbalanceOrg': [1000],
            'newbalanceOrig': [0]}
logger.debug(f'Data for prediction:  {data_aux} ')
predictor = ModelPredictor("./models/logistic_regression_output.pkl")
X_test = onlinefraud_data_pipeline.PIPELINE.fit_transform(pd.DataFrame(data_aux))
print(X_test)
prediction_result = predictor.predict(X_test)
logger.debug(f'Prediction for fraud:  {prediction_result} ')
