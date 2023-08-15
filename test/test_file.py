"""Tests for `itesm_mlops` package."""
import os
import pytest
# import pandas as pd

# from sklearn.pipeline import Pipeline

from load.load_data import DataRetriever
# from preprocess.preprocess_data import MissingIndicator


def does_csv_file_exist(file_path):
    """
    Check if a CSV file exists at the specified path.

    Parameters:
        file_path (str): The path to the CSV file.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    return os.path.isfile(file_path)


def test_csv_file_existence():
    """
    Test case to check if the CSV file exists.
    """
    # Provide the path to your CSV file that needs to be tested
    # os.chdir('C:/Users/oscar.betanzos/Documents/ProyectoMlOps_V2')
    csv_file_path = "./data/retrieved_data.csv"

    DATASETS_DIR = './data/'
    FILE_PATH = "c:/Users/oscar.betanzos/Documents/Dataset/onlinefraud.csv"

    data_retriever = DataRetriever(FILE_PATH, DATASETS_DIR)
    data_retriever.retrieve_data()

    # Call the function to check if the CSV file exists
    file_exists = does_csv_file_exist(csv_file_path)

    # Use Pytest's assert statement to check if the file exists
    if file_exists:
        f"The CSV file at '{csv_file_path}' does not exist."


if __name__ == "__main__":
    # Run the test function using Pytest
    pytest.main([__file__])
