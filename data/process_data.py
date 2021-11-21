import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    This function loads the content of messages and categories datafiles into a Pandas Dataframe.
    Input:
    - messages_filepath(String): location of messages.csv file
    - categories_filepath(String): location of categories.csv file
    Output:
    - df(Dataframe): messages and categories tables merged on 'id'
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,on='id')
    return df

def clean_data(df):
    """
    This function cleans the Dataframe containing messages and categories, 
    performs some transformation to get it ready for ML pipeline.
    Input:
    - df(Dataframe): the Dataframe containing messages and categories
    Output:
    - df(Dataframe): the cleaned and transformed Dataframe from input
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [x[0:len(x)-2] for x in row]
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1
    categories = categories.astype(str)
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1:] if x[-1:]=='0' or x[-1:]=='1' else '0')
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # Replace categories column in df with new category columns
    # drop the original categories column from `df`
    df.drop('categories', inplace=True, axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df = df.drop_duplicates()
    
    return df

def save_data(df, database_filename):
    """
    This function saves the cleaned Dataframe 'df' into a SQLite database file. 
    Input:
    - df(Dataframe): the cleaned and transformed Dataframe containing messages and categories
    - database_filename(String): location to store the database file
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('mess', engine, index=False)
    return

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