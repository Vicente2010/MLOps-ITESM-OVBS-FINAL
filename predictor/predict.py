import joblib
from utilities.logging_util import LoggingSetter

# SETTING THE LOGGER UTILITY
# Using the logger utility for handling log setting and creation
loggerSet = LoggingSetter(__name__)

# All logs will be saved at utilities folder
logger = loggerSet.setting_log('utilities/main.log')


class ModelPredictor:
    """
    A class to load a trained machine learning model and make predictions on new data.

    Parameters:
        model_path (str): Path to the trained model file (joblib format).

    Methods:
        predict(new_data):
            Makes predictions on the provided new_data using the loaded model.

    Usage:
        $ python model_predictor.py trained_models/logistic_regression_output.pkl path_to_new_data
    """

    def __init__(self, model_path):
        """
        Initializes the ModelPredictor instance.

        Parameters:
            model_path (str): Path to the trained model file (joblib format).
        """
        self.model = joblib.load(model_path)
        logger.debug("Model loaded!")

    def predict(self, new_data):
        """
        Makes predictions on the provided new_data using the loaded model.

        Parameters:
            new_data: The data on which to make predictions.

        Returns:
            Predicted outputs from the model.
        """
        logger.debug("Accesing prediction")
        return self.model.predict(new_data)
