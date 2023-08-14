# This file is still in building
# Main module
from load.load_data import DataRetriever
import pandas as pd
# import os

DATASETS_DIR = './data/'
RETRIEVED_DATA = 'retrieved_data.csv'

# The file is too big, in this version we use a local path
# URL = "https://drive.google.com/file/d/1sU4a5BZsBBDT8wqSxCXa-KQcfjS6REIX/view?usp=sharing"
FILE_PATH = "c:/Users/oscar.betanzos/Documents/Dataset/onlinefraud.csv"

# print(os.getcwd())
# Retrieve data
data_retriever = DataRetriever(FILE_PATH, DATASETS_DIR)
result = data_retriever.retrieve_data()
print(result)

# Read data
data = pd.read_csv(DATASETS_DIR + RETRIEVED_DATA)
print(data.head())
