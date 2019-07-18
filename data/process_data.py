import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(left=messages, right=categories, on=["id"])
    return df

def clean_data(df):
    categories_temp = df.categories.str.split(";", expand=True)
    row = categories_temp.iloc[0]
    category_colnames = row.apply(lambda x : re.sub("-\d$", "", x))
    categories_temp.columns = category_colnames
    for column in categories_temp:
        # set each value to be the last character of the string
        categories_temp[column] = categories_temp[column].str.extract(r"(\d)$")
    
        # convert column from string to numeric
        categories_temp[column] = pd.to_numeric(categories_temp[column])
    df = df.drop(["categories"], axis=1)
    df[categories_temp.columns] = categories_temp
    df = df.drop_duplicates()
    return df

def save_data(df, database_filename):
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('CategorizedMessages', engine, if_exists="replace", index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()