# import libraries
import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier


import pickle
nltk.download('stopwords')


def load_data(database_filepath):
    """
    Load data
    
    Input:
        database_filename: Name of the database file, e.g. disaster_response.db
        
    Output:
        X: Messages - features values
        Y: Categories - target values, being the values you want to try to predict
        category_names: Labels of the categories

    """
    
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)

    #Redefine table name as done in the process_data file
    #table = database_filepath.replace(".db","") + "_table"
    
    df = pd.read_sql_table('DisasterResponseTable', engine)

    # define X
    X = df['message']

    # define Y - skip the first four columns id, message, original and genre
    Y = df[df.columns[4:]]

    # define category_names as they will be used within the function evaluate_model
    category_names = list(np.array(Y.columns))

    return X , Y, category_names

def tokenize(text):
    """
    Clean, normalize, tokenize and lemmonize the text
    
    Input:
        text: text that needs to bei transformed
        
    Output:
        token_final: cleaned, normalizend and lemmonized text tokens

    """
    
    
    # define url_regex as the regular expression to detect a url
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")


    # Normalize text:
    # Remove punctuation characters and 
    # Convert to lowercase in one step
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize text
    # Split text into words using NLTK word_tokenize
    token = word_tokenize(text)

    # Remove stop words using NLTK stopwords
    token = [w for w in token if w not in stopwords.words("english")]

    # Reduce words to their root form using NLTK WordNetLemmatizer
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in token]

    # Lemmatize verbs by specifying pos
    token_final = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]

    return token_final


def build_model():
    """
    Definition of the pipeline using CountVectorizer, TfidfTransformer, 
    RandomForestClassifier and GridSearch
    
    Input: 
        None
        
    Output:
        cv: GridSearch result

    """
    # build pipeline
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42)))
    ])
    
    
    # Improved parameters 
    parameters = {'clf__estimator__n_estimators':[5]
                #'clf__estimator__n_estimators': [20, 50],
               # 'clf__estimator__min_samples_split': [2, 4],
                #'clf__estimator__max_features': ['sqrt', 'log2']
                }



    # new model with improved parameters
    cv = GridSearchCV(pipeline, param_grid=parameters, cv = 2)
   
    return cv
    

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Definition of the pipeline using CountVectorizer, TfidfTransformer, 
    RandomForestClassifier and GridSearch
    
    Input: 
        model: Machine Learning Model
        X_test: Test messages 
        Y_test: Categories of the test messages
        category_names: Labels of the categories
        
    Output:
        Y_pred: Categories of the X_test data based upon the trained data obtained from the model
        class_report: printed classification report 

    """
    Y_pred = model.predict(X_test)

    #print the classification report
    print(classification_report(Y_test.values, Y_pred, target_names=Y_test.columns.values, zero_division=0))


def save_model(model, model_filepath):
    """
    Save the model as a pickle file 
    Input: 
        model: Machine Learning Model
        model_filepath: path of the generated pickle file
        
    Output:
        A pickle file of saved model
    """
    
    pickle.dump(model, open(model_filepath, "wb"))



def main():
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        print('Building model...')
        model = build_model()
           
        print('Training model...')
        model.fit(X_train, Y_train)
           
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
           
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
           
        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()