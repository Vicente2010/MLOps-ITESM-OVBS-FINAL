from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from preprocess.preprocess_data import (
    OneHotEncoder,
    FeatureSelector,
)
from utilities.logging_util import LoggingSetter

# SETTING THE LOGGER UTILITY
# Using the logger utility for handling log setting and creation
loggerSet = LoggingSetter(__name__)

# All logs will be saved at utilities folder
logger = loggerSet.setting_log('utilities/main.log')


class OnlineFraudPipeline:
    """
    A class representing the Titanic data processing and modeling pipeline.

    Attributes:
        NUMERICAL_VARS (list): A list of numerical variables in the dataset.
        CATEGORICAL_VARS_WITH_NA (list): A list of categorical variables with missing values.
        NUMERICAL_VARS_WITH_NA (list): A list of numerical variables with missing values.
        CATEGORICAL_VARS (list): A list of categorical variables in the dataset.
        SEED_MODEL (int): A seed value for reproducibility.

    Methods:
        create_pipeline(): Create and return the Titanic data processing pipeline.
    """

    def __init__(self, seed_model, categorical_vars, selected_features, selected_features_dt):
        self.SEED_MODEL = seed_model
        # self.NUMERICAL_VARS = numerical_vars
        # self.CATEGORICAL_VARS_WITH_NA = categorical_vars_with_na
        # self.NUMERICAL_VARS_WITH_NA = numerical_vars_with_na
        self.CATEGORICAL_VARS = categorical_vars
        self.SELECTED_FEATURES = selected_features
        self.SELECTED_FEATURES_DT = selected_features_dt

    def create_pipeline(self):
        """
        Create and return the Titanic data processing pipeline.

        Returns:
            Pipeline: A scikit-learn pipeline for data processing and modeling.
        """
        self.PIPELINE = Pipeline(
            [
                                ('dummy_vars', OneHotEncoder(variables=self.CATEGORICAL_VARS)),
                                ('feature_selector', FeatureSelector(self.SELECTED_FEATURES)),
                                ('scaling', MinMaxScaler()),
            ]
        )
        logger.debug("Creating pipeline")
        return self.PIPELINE

    def fit_logistic_regression(self, X_train, y_train):
        """
        Fit a Logistic Regression model using the predefined data preprocessing pipeline.

        Parameters:
        - X_train (pandas.DataFrame or numpy.ndarray): The training input data.
        - y_train (pandas.Series or numpy.ndarray): The target values for training.

        Returns:
        - logistic_regression_model (LogisticRegression): The fitted Logistic Regression model.
        """
        logistic_regression = LogisticRegression(C=0.0005, class_weight='balanced', random_state=self.SEED_MODEL)
        pipeline = self.create_pipeline()
        pipeline.fit(X_train, y_train)
        logistic_regression.fit(pipeline.transform(X_train), y_train)
        logger.debug("Training Logistic Regresion")
        return logistic_regression

    def transform_test_data(self, X_test):
        """
        Apply the data preprocessing pipeline on the test data.

        Parameters:
        - X_test (pandas.DataFrame or numpy.ndarray): The test input data.

        Returns:
        - transformed_data (pandas.DataFrame or numpy.ndarray): The preprocessed test data.
        """
        pipeline = self.create_pipeline()
        return pipeline.transform(X_test)

    def fit_decision_tree(self, X_train, y_train):
        """
        Fit a Decision Tree model using the predefined data preprocessing pipeline.

        Parameters:
        - X_train (pandas.DataFrame or numpy.ndarray): The training input data.
        - y_train (pandas.Series or numpy.ndarray): The target values for training.

        Returns:
        - logistic_regression_model (LogisticRegression): The fitted Logistic Regression model.
        """
        decisionTree = DecisionTreeClassifier()
        pipeline_dt = self.create_DTpipeline()
        pipeline_dt.fit(X_train, y_train)
        decisionTree.fit(pipeline_dt.transform(X_train), y_train)
        logger.debug("Training Decision Tree")
        return decisionTree

    def create_DTpipeline(self):
        """
        Create and return the Online Fraud data processing pipeline for DecisionTree.

        Returns:
            Pipeline: A scikit-learn pipeline for data processing and modeling.
        """
        self.PIPELINE_DT = Pipeline([('feature_selector', FeatureSelector(self.SELECTED_FEATURES_DT)),])
        logger.debug(f"Creating DecisionTree Pipeline, Selected Features: {self.SELECTED_FEATURES_DT}")
        return self.PIPELINE_DT
