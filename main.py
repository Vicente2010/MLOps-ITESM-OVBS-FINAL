# This file is still in building
# Main module
from load.load_data import DataRetriever
import pandas as pd
import joblib
from train.train_data import OnlineFraudPipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from fastapi import FastAPI
# Body,
# Path,
# Query,
# HTTPException,
# status

from utilities.logging_util import LoggingSetter
# import os

# SETTING THE LOGGER UTILITY
# Using the logger utility for handling log setting and creation
loggerSet = LoggingSetter(__name__)

# All logs will be saved at utilities folder
logger = loggerSet.setting_log('utilities/main.log')

# Constants for the load process
# The file is too big, in this version we use a local path instead of an URL
# URL = "https://drive.google.com/file/d/1sU4a5BZsBBDT8wqSxCXa-KQcfjS6REIX/view?usp=sharing"
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
data_retriever = DataRetriever(FILE_PATH, DATASETS_DIR)
result = data_retriever.retrieve_data()
logger.info(f"Result of DataRetriever.data_retreiver:  {result}")
# print(result)

# Read data
data = pd.read_csv(DATASETS_DIR + RETRIEVED_DATA)

# Cambiar el logging por logger
logger.info(f"Data head: \n {data.head()}")

# Instantiate the Pipeline class
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
print(f'test roc-auc : {roc_auc_score(y_test, proba_pred)}')
print(f'test accuracy: {accuracy_score(y_test, class_pred)}')

# Displaying confusion matrix
conf_mx = confusion_matrix(y_test, y_pred, labels=logistic_regression_model.classes_)
print(conf_mx)

# Save the model using joblib
save_path = TRAINED_MODEL_DIR + PIPELINE_SAVE_FILE
joblib.dump(logistic_regression_model, save_path)
print(f"Model saved in {save_path}")

# API predict test
data_aux = {'type': ['CASH_OUT'],
            'amount': [1000],
            'oldbalanceOrg': [1000],
            'newbalanceOrig': [0]}
print(data_aux)
X_test = onlinefraud_data_pipeline.PIPELINE.fit_transform(pd.DataFrame(data_aux))
print(X_test)
print(logistic_regression_model.predict(X_test))

# API code

app = FastAPI()

"""
PARAMETER VALUES
Values are required after de endpoint.
"""


@app.get('/', status_code=200)
async def healthcheck():
    return 'Online Fraud Predictor is all ready to go!'


@app.get("/predict/")
async def predict(v_type: str = "CASH_OUT", v_amount: float = 1.0, v_oldbalanceOrig: float = 1.0, v_newbalanceOrig: float = 1.0):
    """
    Makes a prediction of fraud or not fraud given the transaction characteristics

    This endpoint calculates a prediction of fraud given four variables of a transaction and returns the result.

    Parameters:
    - **type**: The first float value. (Default: 1.0)
    - **amoun**: The second float value. (Default: 2.0)

    Responses:
    - **200 OK**: The subtraction was calculated successfully.
        - Response JSON:
            ```
            {
                "resultado": srt (subtraction of v1 and v2)
            }
            ```
    - **422 Unprocessable Entity**: Validation error.
        - Response JSON:
            ```
            {
                "detail": [
                    {
                        "loc": ["query", "v1"],
                        "msg": "value is not a valid float",
                        "type": "type_error.float"
                    },
                    {
                        "loc": ["query", "v2"],
                        "msg": "value is not a valid float",
                        "type": "type_error.float"
                    }
                ]
            }
            ```
    """

    # initialize list of lists
    # data_aux = [['type', v_type], ['amount', v_amount], ['oldbalanceOrg', v_oldbalanceOrig], ['newbalanceOrg', v_newbalanceOrig]]
    data_aux = {'type': [v_type],
                'amount': [v_amount],
                'oldbalanceOrg': [v_oldbalanceOrig],
                'newbalanceOrig': [v_newbalanceOrig]}

    X_test = onlinefraud_data_pipeline.PIPELINE.fit_transform(pd.DataFrame(data_aux))

    result = str(logistic_regression_model.predict(X_test))
    print(f"'resultado': {result}")
    print(f"'resultado': {result}")
    return {"resultado": result}
