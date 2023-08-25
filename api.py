from fastapi import FastAPI
from utilities.logging_util import LoggingSetter
from predictor.predict import ModelPredictor
from main import onlinefraud_data_pipeline
import pandas as pd
from starlette.responses import JSONResponse
from models.models import OnlineFraud

# SETTING THE LOGGER UTILITY
# Using the logger utility for handling log setting and creation
loggerSet = LoggingSetter(__name__)

# All logs will be saved at utilities folder
logger = loggerSet.setting_log('utilities/main.log')


# API code
app = FastAPI()

"""
PARAMETER VALUES
Values are required after de endpoint.
"""


@app.get('/', status_code=200)
async def healthcheck():
    logger.debug('Online Fraud Predictor is all ready to go!')
    return 'Online Fraud Predictor is all ready to go!'


@app.get("/predict/")
async def predict(v_type: str = "CASH_OUT", v_amount: float = 1.0, v_oldbalanceOrig: float = 1.0, v_newbalanceOrig: float = 1.0):
    """
    Makes a prediction of fraud or not fraud given the transaction characteristics

    This endpoint calculates a prediction of fraud given four variables of a transaction and returns the result.

    Parameters:
    - **type**: The first float value. (Default: 1.0)
    - **amount**: The second float value. (Default: 2.0)

    """

    # initialize list of lists
    # data_aux = [['type', v_type], ['amount', v_amount], ['oldbalanceOrg', v_oldbalanceOrig], ['newbalanceOrg', v_newbalanceOrig]]
    data_aux = {'type': [v_type],
                'amount': [v_amount],
                'oldbalanceOrg': [v_oldbalanceOrig],
                'newbalanceOrig': [v_newbalanceOrig]}

    X_test = onlinefraud_data_pipeline.PIPELINE.fit_transform(pd.DataFrame(data_aux))
    predictor = ModelPredictor("./models/logistic_regression_output.pkl")
    result = str(predictor.predict(X_test))
    print(f"'Prediction for online fraud': {result}")
    print(f"'Prediction for online fraud': {result}")
    logger.debug(f"'Prediction for online fraud': {result}")
    return {"Prediction for online fraud": result}


@app.post("/predict_fraud/")
def predict_fraud(onlinefraud_features: OnlineFraud) -> JSONResponse:
    predictor = ModelPredictor("./models/logistic_regression_output.pkl")
    X = [onlinefraud_features.type,
         onlinefraud_features.amount,
         onlinefraud_features.oldbalanceOrg,
         onlinefraud_features.newbalanceOrig]

    logger.debug(f"Input values: {[X]} , X Type: {type([X])} ")

    data_aux = {'type': onlinefraud_features.type,
                'amount': onlinefraud_features.amount,
                'oldbalanceOrg': onlinefraud_features.oldbalanceOrg,
                'newbalanceOrig': onlinefraud_features.newbalanceOrig}

    logger.debug(f"Input values: {print(data_aux)}")

    X_test = onlinefraud_data_pipeline.PIPELINE.fit_transform(pd.DataFrame(data_aux, index=[0]))
    prediction = predictor.predict(X_test)

    logger.debug(f"Prediction for online fraud: {prediction}")
    return JSONResponse(f"Prediction for online fraud: {prediction}")


@app.post("/predict_fraud_dt/")
def predict_fraud_dt(onlinefraud_features: OnlineFraud) -> JSONResponse:
    predictor = ModelPredictor("./models/decision_tree_output.pkl")
    X = [onlinefraud_features.type,
         onlinefraud_features.amount,
         onlinefraud_features.oldbalanceOrg,
         onlinefraud_features.newbalanceOrig]

    logger.debug(f"Input values: {[X]} , X Type: {type([X])} ")

    data_aux = {'amount': onlinefraud_features.amount,
                'oldbalanceOrg': onlinefraud_features.oldbalanceOrg,
                'newbalanceOrig': onlinefraud_features.newbalanceOrig}

    logger.debug(f"Input values: {print(data_aux)}")

    X_test = onlinefraud_data_pipeline.PIPELINE_DT.fit_transform(pd.DataFrame(data_aux, index=[0]))
    prediction = predictor.predict(X_test)

    logger.debug(f"Prediction for online fraud: {prediction}")
    return JSONResponse(f"Prediction for online fraud: {prediction}")
