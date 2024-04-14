# Project Summary: Disaster Message Classification Web App

In this project, we work with a dataset containing real messages sent during disaster events. Our goal is to create a robust machine learning pipeline that can effectively categorize these messages. By doing so, we can route them to the appropriate disaster relief agencies for timely assistance.

## Key Components of the Project:

### Data Exploration and Preprocessing:
We’ll analyze the dataset, handle missing values, and preprocess the text messages.
Feature engineering may involve extracting relevant information from the messages.
### Machine Learning Pipeline:
We’ll build a classification model that can predict the category of each message.
The pipeline will include tokenization, feature extraction, model training, and evaluation.
### Web App Development:
We’ll create a user-friendly web app where emergency workers can input new messages.
The app will process the input and display classification results across several predefined categories.
### Visualization:
The web app will also provide visualizations of the dataset, helping users understand patterns and trends.
Graphs, charts, and summary statistics will enhance data exploration.
Overall, this project aims to streamline disaster response efforts by automating message categorization and providing actionable insights through an intuitive web interface.

Your project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data


# Disaster Response Pipeline Project
There are 3 main folders in this repo,
1. app-> it has fast api code
2. data-> it has code related to data preprocessing
3. models-> it has model infer code

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

