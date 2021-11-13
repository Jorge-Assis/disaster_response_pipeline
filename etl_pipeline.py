def clean_data(df):
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
    
def load_data(data_file):
    # read in file
    df = pd.read_csv(data_file)

    # clean data
    df = clean_data(df)
    
    # load to database
    engine = create_engine('sqlite:///messages.db')
    df.to_sql('messages', engine, index=False)

    # define features and label arrays
    X = df["message"]
    y = np.asarray(df[df.columns[4:]])
    
    return X, y


def build_model():
    # text processing and model pipeline


    # define parameters for GridSearchCV


    # create gridsearch object and return as final model pipeline


    return model_pipeline


def train(X, y, model):
    # train test split


    # fit model


    # output model test results


    return model


def export_model(model):
    # Export model as a pickle file



def run_pipeline(data_file):
    X, y = load_data(data_file)  # run ETL pipeline
    model = build_model()  # build model pipeline
    model = train(X, y, model)  # train model pipeline
    export_model(model)  # save model


if __name__ == '__main__':
    data_file = sys.argv[1]  # get filename of dataset
    run_pipeline(data_file)  # run data pipeline