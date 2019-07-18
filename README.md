# Disaster Response Pipeline Project
Team at Figure Eight provided pre-labeled tweets and text messages from real world disasters. In this project I build an ETL pipeline to prepare the data and use it in a Machine Learning Pipeline to build a supervised learning model to categorize messages into different scenarios that the disaster response team may handle effectively.

## Problem
In real life disasters, millions of text messages and tweets are received either directly or via social media right at the time when the disaster response organizations have the least capacity to filter and then pull out the messages which are the most important. There might one on every thousands of messages that might be relevant to the disaster response professionals. 

At the time of disaster, different organizations take care of different parts of the problem. For e.g. one organization may care about water, another would take care about blocked roads and some other would take care of medical supplies etc. In the dataset, these kind of categories are pulled out from these datasets. The datasets from different disasters have been combined and consistently labeled with categories with help of Figure Eight, the human and machine learning enabled data annotation service.

Talking about the nature of the messages and the category assigned to them, it is possible that the exact word that describes the category for e.g. 'water' may not be present in the text of the message and that category still needs to be inferred based on the presence of other relevant words and phrases for e.g. 'thirsty'. Therefore, a simple keyword matching may not be effective way to categorize the messages. Supervised machine learning model can be more helpful to tackle this challenge effectively.

## Solution
- ETL pipeline loads pre-labeled dataset provided by Figure Eight team and transforms the dataframe such that the categories assigned to each message is indicated by value of 1 or 0 in a separate column for that category.  The dataframe is stored in table of SQLite database.
- The Natural Language Processing enabled Machine Learning Pipeline reads the dataframe from SQLite database. It transforms the text message into tokens and then the tokens are converted to tf-idf vectors. These numerical variables are used as input variables for training supervised machine learning model. The columns for the categories in the dataframe are used by supervised machine learning model as target variable for prediction of category for newly received message. The learnt statistical model is saved in the pickle file format.
- A web application created using Flas is able to take any message as input and is able to predict the suitable categories for that message based on its text content with help of the statistical model trained by the ML pipeline.

## Project Structure

    - app    
    | - template    
    | |- master.html # main page of web app    
    | |- go.html # classification result page of web app    
    |- run.py # Flask file that runs app     
    
    - data    
    |- disaster_categories.csv # data to process    
    |- disaster_messages.csv # data to process    
    |- process_data.py # ETL pipeline    
    
    - models    
    |- train_classifier.py # NLP & ML pipeline
    
    - README.md

### Instructions to Run project
1. The project depends on several libraries including plotly, pandas, nltk, flask, sklearn, sqlalchemy, numpy, re, string, pickle
2. Run the following commands in the project's root directory to set up SQLite database and Supervised machine learning model to categorize messages.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Open http://0.0.0.0:3001/ in browser
