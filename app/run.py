import json
import plotly
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

nltk.download('stopwords')
stop_words = stopwords.words('english')

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [word for word in tokens if word not in stop_words]
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    graph_data = []
    genre_counts_df = df.groupby('genre').count()
    
    for col in genre_counts_df.columns:
        
        genre_names = list(genre_counts_df[col].index)
        genre_counts = genre_counts_df[col].values
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
        graphs = {
                'data': [
                    Bar(
                        x=genre_names,
                        y=genre_counts
                    )
                ],

                'layout': {
                    'title': f'Distribution of {col} Genres',
                    'yaxis': {
                        'title': "Count"
                    },
                    'xaxis': {
                        'title': "Genre"
                    }
                }
            }
        graph_data.append(graphs)
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graph_data)]
    graphJSON = json.dumps(graph_data, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


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
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()