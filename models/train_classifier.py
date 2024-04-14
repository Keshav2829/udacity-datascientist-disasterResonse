import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

nltk.download('wordnet')
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def load_data(database_filepath):
    """Load data from sql database and create Feature and Target dataframes
    Input: database path
    Output: feature dataframe, target dataframe, classes"""
    # engine = create_engine('sqlite:///disaster_msg.db')
    df = pd.read_sql_table('messages',"sqlite:///"+ database_filepath)
    X = df['message']
    y_col = [col for col in df.columns if col not in ['message','original', 'genre', 'id']]
    Y = df[y_col]

    return X,Y,y_col


def tokenize(text):
    """Prepross text using tokenization and lemmatization
    Input: input text
    Output: tokenized list of word"""
    # convert to lower case
    text = text.lower()
    # tokenization
    text_tokens = word_tokenize(text)

    # remove stop words

    text_tokens = [word for word in text_tokens if word not in stop_words]
    #lamitization
    text_tokens =  [lemmatizer.lemmatize(word) for word in text_tokens]

    return text_tokens


def build_model():
    """Prepare Model using Sklearn Pipeline and grid search
    Input: None
    Output: Model"""
    pipeline = Pipeline([('text_pipline', Pipeline([
                                            ('vect', CountVectorizer(tokenizer= tokenize)),
                                            ('tfidf', TfidfTransformer())])),
                    ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    parameters = {'clf__estimator__n_estimators' :[50,100,150],
                  'clf__estimator__max_depth' : [None, 5,10]}
    
    model = GridSearchCV(pipeline, parameters, cv=5)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate preformance of the model
    Input: model, testinput, test target, classes"""
    y_pred = model.predict(X_test)
    for index, column in enumerate(Y_test):
        print(column, classification_report(Y_test[column], y_pred[:, index]))


def save_model(model, model_filepath):
    """Save the model in pickle file
    Input: Model, filepath"""
    pickle.dump(model, open(model_filepath, 'wb'))


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