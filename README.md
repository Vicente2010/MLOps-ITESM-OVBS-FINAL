# ITESM-MLOps-Project
This repository contains the files related to first part of the final proyect 

Student ID : A01688409

Student Name : Oscar Vicente Betanzos Sánchez

Teacher: Dr. Carlos Noé López Mejía

This project uses the following dataset : 

https://www.kaggle.com/datasets/jainilcoder/online-payment-fraud-detection

This project uses the following notebook for the reference notebook

https://www.kaggle.com/code/khizarsultan/fraud-detection

# About the dataset: 
This dataset contains historical information about fraudulent transactions which can be used to detect fraud in online payments

**Features list**

**step:** represents a unit of time where 1 step equals 1 hour

**type:** type of online transaction

**amount:** the amount of the transaction

**nameOrig:** customer starting the transaction

**oldbalanceOrg:** balance before the transaction

**newbalanceOrig:** balance after the transaction

**nameDest:** recipient of the transaction

**oldbalanceDest:** initial balance of recipient before the transaction

**newbalanceDest:** the new balance of recipient after the transaction

**isFraud:** fraud transaction 



# Objective/Scope

The objective of this project is to train a machine learning model for classifying fraudulent and non-fraudulent payments using features available in the current dataset.

The current scope is limited to have the proper structure for a MLOps project with a few functionalities, modular structure and to have a simple ML model to solve the objective.

This project is basically a *Proof of Concept* for testing the proper structure of a MLOps project

# About the project requirements and venv
This project was developed using Python 3.10.9 .

Check the **requirements.txt** file for information on the packages' versions used in the project.

Check the **requirements_extended.txt** file for information on all the packages' versions used in the development.

To create virtual environment and install requirements use the following lines

```
py -3.10 -m venv venv-proy01

.\venv-proy01\Scripts\activate

pip install -r requirements.txt
```

# Main file

The main file named *main.py* has all the functions needed to solve the objective. 

This file uses funcions from the modules in the project folders.

## Load Module

This module keeps the processes to import the dataset. 

**NOTE:** The current scope of this project doesn't reach the functionality to download the dataset and store it in the *data* folder because the file is 470MB.

The current functionality of this project is :
 
1. Download the dataset from the url from the https://www.kaggle.com/datasets/jainilcoder/online-payment-fraud-detection

1. Copy you local path to the dataset in the variable **FILE_PATH** in the *main.py* file

1. After using the load functionality in the *main.py* file , delete the dataset from the data folder in order to avoid commiting with file in the project. This project can't handle large files.

## Preprocess

This module keeps the transformations done to the dataset.

## Train 

This module holds the Pipeline to preprocess the dataset and the models methods that are going to be trained with the data.

## Models 

The main file export the trained model in this file to keep the results.

## Test

This module holds test to validate the functionalities in the other modules.

It has its own ReadMe

## Notebook

This folder keeps the original notebook

