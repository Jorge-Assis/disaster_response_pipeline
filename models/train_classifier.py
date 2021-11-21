import sys
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import pickle

def load_data(database_filepath):
    """
    This function loads the content of messages and categories table
    from SQLite database into variables X and y.
    Input:
    - database_filepath(String): location of the database file
    Output:
    - X(Dataframe): input messages
    - y(Dataframe): message labels
    - category_names(list): list of category names for messages
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('mess',engine)
    X = df.iloc[:,1]
    y = df.iloc[:,4:]
    category_names = y.columns
    return X, y, category_names

def tokenize(text):
    """
    This function tokenizes the text, lemmatizes it, and change text to lower case.
    Input:
    - text(String): input text
    Output:
    - clean_tokens(String): tokenized and lemmatized text
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    """
    This function defines a ML pipeline with parameters found using GridSearch.
    (GridSearch part is done in the attached Jupyter notebook)
    Output:
    - pipeline(Pipeline): ML pipeline
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=200,min_samples_leaf=10,max_features=0.5,n_jobs=-1)))
    ])

    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function evaluates the ML models using F1 score.
    Input:
    - model: a ML model
    - X_test(Dataframe): the test data
    - Y_test(Dataframe): the test labels
    - category_names(list): list of category names for messages
    """
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred,target_names=category_names))
    return

def save_model(model, model_filepath):
    """
    This function saves the trained model inpto a pickle file
    for production.
    Input:
    - model: a ML model
    - model_filepath: location to save the model
    """
    with open(model_filepath, 'wb') as model_file:
        pickle.dump(model, model_file)
    return

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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