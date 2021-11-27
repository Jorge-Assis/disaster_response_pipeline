import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from flask import Markup
from plotly.graph_objs import Bar
import joblib
import logging
from sqlalchemy import create_engine

# App config.
DEBUG = True
app = Flask(__name__)

def return_figures(df):
    """Creates plotly visualizations

    Args:
        A pd frame;

    Returns:
        list (dict): list containing plotly visualizations

    """

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    graph1 = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

           'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    df_categories = df[df.columns[4:]]
    category_sum = df_categories.sum()
    categories_names = list(category_sum.index)
    graph2 = [
        {
            'data': [
                Bar(
                    x=categories_names,
                    y=category_sum
                )
            ],

           'layout': {
                'title': 'Distribution of Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        }
    ]

    # ===  Plot the graphs offline with the div format to be rendered on html ===
    Plot_one = plotly.offline.plot(graph1[0], 
            config={"displayModeBar": False}, 
            show_link=False, include_plotlyjs=False, 
            output_type='div')
    Plot_one = Markup(Plot_one)

    Plot_two = plotly.offline.plot(graph2[0], 
            config={"displayModeBar": False}, 
            show_link=False, include_plotlyjs=False, 
            output_type='div')
    Plot_two = Markup(Plot_two)

    # append all charts to the figures list
    plots_list = []
    plots_list.append(Plot_one)
    plots_list.append(Plot_two)

    return plots_list

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('message', engine)

# create visuals
figures = return_figures(df)

# load model
model = joblib.load("./models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # render web page with plotly graphs
    return render_template('master.html', posts=figures)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results,
        posts=figures
    )

@app.errorhandler(500)
def server_error(e):
    logging.exception('some error')
    return """
    And internal error <pre>{}</pre>
    """.format(e), 500

if __name__ == '__main__':
    #main()
    app.run()
