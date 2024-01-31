# Disaster Response Pipeline Project (Udacity - Data Scientist Nanodegree Program)
## by Data Camp 442


## Project Overview

In this project I analyzed the disaster data from Appen (formerly Figure 8) to build a model for an API that classifies disaster messages.

The data set contains real messages that were sent during disaster events. I created a machine learning pipeline to categorize these events so that the messages can be sent to an appropriate disaster relief agency.

Your project also includes a web app where an emergency worker can input a new message and get classification results in several categories. 
The web app will also display visualizations of the data.


## Content

### data
- *disaster_messages.csv:*<br>
  a dataset that contains text messages related to disaster events (provided by Appen)
- *disaster_categories.csv*<br>
  contains information about the categories that each message in the disaster_messages.csv file belongs to.
- *process_data.py*<br>
  ETL pipeline used to load, clean, extract feature and store data in SQLite database
- *ETL Pipeline Preparation.ipynb*<br>
  Jupyter Notebook used to prepare ETL pipeline
- *DisasterResponse.db*<br>
  SQLite database used to store the cleaned and processed data

### models
- *train_classifier.py*<br>
  Python script used to train a machine learning model using the cleaned and preprocessed data from the DisasterResponse.db database
- *ML Pipeline Preparation.ipynb*<br>
  Jupyter Notebook used to prepare ML pipeline


### app
- *run.py*<br>
  Python script to run a web application that allows users to interact with the trained machine learning model and get predictions for new messages
- *Folder: templates*<br>
  web dependency files (go.html & master.html) required to run the web application.

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
The main page includes two visualizations.<br>
<br>
![image](https://github.com/Datacamp442/UDACITY-Data-Scientist-Project-2/assets/154692077/c1a88b17-ac19-4e1f-8748-d279898094cf)
<br>
Message "There is heavy rain"...<br>
<br>
![image](https://github.com/Datacamp442/UDACITY-Data-Scientist-Project-2/assets/154692077/75e6d695-372f-47f6-b069-3b5cdbb68eae)
 <br>
...leads to the categories "Related" and "Weather Related"<br>
<br>
![image](https://github.com/Datacamp442/UDACITY-Data-Scientist-Project-2/assets/154692077/2e294d4f-427e-426c-9795-18be459dc2dc)


