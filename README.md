# Disaster Response Pipeline Project (Udacity - Data Scientist Nanodegree Program)
## by Data Camp 442


## Project Overview

In this project I analyzed the disaster data from Appen (formerly Figure 8) to build a model for an API that classifies disaster messages.

The data set contains real messages that were sent during disaster events. I created a machine learning pipeline to categorize these events so that the messages can be sent to an appropriate disaster relief agency.

Your project also includes a web app where an emergency worker can input a new message and get classification results in several categories. 
The web app will also display visualizations of the data.


## Content

### data
disaster_messages.csv - a dataset that contains text messages related to disaster events (provided by Appen)
disaster_categories.csv - contains information about the categories that each message in the disaster_messages.csv file belongs to.
process_data.py - ETL pipeline used to load, clean, extract feature and store data in SQLite database
ETL Pipeline Preparation.ipynb - Jupyter Notebook used to prepare ETL pipeline
DisasterResponse.db - SQLite database used to store the cleaned and processed data

### models
train_classifier.py - Python script used to train a machine learning model using the cleaned and preprocessed data from the DisasterResponse.db database
classifier.pkl - pickle file of the trained machine learning model that is created by the train_classifier.py script 
ML Pipeline Preparation.ipynb - Jupyter Notebook used to prepare ML pipeline

### app
run.py - Python script to run a web application that allows users to interact with the trained machine learning model and get predictions for new messages
Folder: templates - web dependency files (go.html & master.html) required to run the web application.

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Screenshots
