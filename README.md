# Keeping Track of a Disaster! Assisting Disaster Relief Agencies by Providing Relevant Messages  

### Overview
In this project, we are in a situation when a natural disaster has just happened. People are communicating on social meadia expressing their needs and asking for help. We want to keep tracks of those relevant conversation, identifying people who are in need by categorizing the tweets and text messages sent around that location at that time. For example, we would like to know if a tweet is related to the disaster or not, or if the tweet is about food shortage, water depletion, or a child being alone. The results may then be forwarded to disaster relief agencies so that help can be provided promptly.

### Dataset Description
The dataset we use in this project is released by Figure Eight and is available [online](https://www.figure-eight.com/dataset/combined-disaster-response-data/). The dataset contains 30,000 messages drawn from events including an earthquake in Haiti in 2010, an earthquake in Chile in 2010, floods in Pakistan in 2010, super-storm Sandy in the U.S.A. in 2012, and news articles spanning a large number of years and 100s of different disasters. A screenshot of the messages in the dataset is given below.

![alt text](/figures/disaster-mess.PNG "mess")

Each message is tagged with a the categories it belongs to. We have 36 categories in total.
Example:

![alt text](/figures/disaster-mess-labels.PNG "mess-labels")

For a complete list of message labels, you can refer to the notebook `ETL Pipeline Preparation.ipynb`. The messages and their labels are available in `data/disaster_messages.csv` and `data/disaster_categories.csv`, respectively.

### System Model
One challenge in this project is the fact that we have no simple way to filter the messages about disasters only. For example, we would like to detect a tweet like this 

> We are 36 people at the ABC church without clean water. Please help!

If we simply filter text by keywords such as `water`, we may end up with a tweet such as

> Craving water after the basketball match today. Great jobs team.

Therefore, we would like to build classifier for the text messages, giving out categories that the text may belong to so that disaster relief agencies can act promptly on the right people who need help. We further deploy our model to a web app so that everyone can easily copy the message into our classifier to receive the result. The project is separated into three parts.

1. **ETL Pipeline**
- Load the dataset files, combine them to get the correct labels for each message
- Encode the labels, remove duplicated entries
- Store the cleaned dataset into a database file

2. **Machine Learning Pipeline**
- Load data from database
- Design ML pipeline: tokenize the messages -> tranform using TF-IDF -> input to multi-output Random Forest Classifier
- Use GridSearch to optimize parameters
- Output final model for web-deployment

The final model achieves a **F1 score of 0.68**.

3. **Web Deployment**
- Design backend using `Flask`, visualization using `plotly`
- Design frontend using `Bootstrap`  

Webapp screenshots:

![alt text](/figures/disaster-web1.PNG "mess-labels")
![alt text](/figures/disaster-web2.PNG "mess-labels")

### Build Instructions
In this project folder, `ETL Pipeline Preparation.ipynb` and `ML Pipeline Preparation.ipynb` Jupyter notebooks are for experimenting the steps we will take to build a ETL pipeline to process the data and an ML pipeline to build the classifier. After that, we moves the codes to the python files to run the web app. To run the web app, follow the steps listed below.

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ or http://localhost:3001/


