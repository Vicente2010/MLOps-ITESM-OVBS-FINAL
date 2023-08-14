# ITESM-MLOps-Project
This repository contains the files related for the final proyect 

Student ID : A01688409

Student Name : Oscar Vicente Betanzos Sánchez

Teacher: Dr. Carlos Noé López Mejía

This project uses the following dataset : 

https://www.kaggle.com/datasets/jainilcoder/online-payment-fraud-detection

This project uses the following notebook for the reference notebook

https://www.kaggle.com/code/khizarsultan/fraud-detection

# About the dataset: 
This dataset contains historical information about fraudulent transactions which can be used to detect fraud in online payments

Features Available:

step: represents a unit of time where 1 step equals 1 hour

type: type of online transaction

amount: the amount of the transaction

nameOrig: customer starting the transaction

oldbalanceOrg: balance before the transaction

newbalanceOrig: balance after the transaction

nameDest: recipient of the transaction

oldbalanceDest: initial balance of recipient before the transaction

newbalanceDest: the new balance of recipient after the transaction

isFraud: fraud transaction 


# Objective

To train a machine learning model for classifying fraudulent and non-fraudulent payments using features available.

# About the project requirements and venv
This project was developed using Python 3.10.9 .

Check the requirements.txt file for information on the packages' versions used in the project.

Check the requirements_extended.txt file for information on all the packages' versions used in the development.

To create virtual environment and install requirements use the following lines

py -3.10 -m venv venv-proy01

.\venv-proy01\Scripts\activate

pip install -r requirements.txt
