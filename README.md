# thread_collector
A project that scrapes, cleans, sorts, categorises and creates url for tweets, specifically threads on twitter.
It contains the following files:

### * * grab_data.py

The first, grab_data.py file is designed to scrape tweets that are related to or have the characters "thread or 🧵" by using the Tweepy library. The code in this file, grab_data.py, provides the necessary authentication keys and handles the query for the tweets. It also collects the tweets, stores them in a CSV file and prints out the number of tweets scraped and the time taken to scrape. The data collected can then be used for further analysis and insights.

### * * clean_data.py

This second file, clean_data.py, is designed to filter tweets according to the criteria specified by the user. The code in the file reads data from a CSV file, finds tweets with a retweet count above 700 and containing either thread or 🧵 in the text, sorts the results by size of retweet count, drops all duplicate tweets, and creates a new file with the set of tweets that meet the criteria.

### * * model.py

The third file, model.py, uses natural language processing and machine learning to categorize the tweets from previously cleaned data. This file takes the cleaned tweets and classifies them by news category using data from varied sources which are then used to build a model that is used to categorize the text from each tweet. The output of this process is a new file containing the cleaned tweets with news categories assigned to them. This provides a better insight into the topics that people are discussing and allows for further analysis of the data.

### * * create_url.py

The fourth file, create_url.py, is used to create URLs that direct to a tweet given its user name and tweet_id. This file reads data from the previously created CSV file of categorised tweets, iterates through each row and creates a furl using the user name and tweet_id. The links are then appended to the categorised_tweets DataFrame and reindexed. Finally, the new DataFrame containing the links is saved as a new CSV file, weekly_tweets.csv. This file is useful for quickly accessing the tweets without having to go through long search processes

### * * main.py

The fifth file, main.py, is the main file used to execute the other four files in a specified order. The list of the file names is defined in the beginning of the file and is iterated through using a for loop with each file name being passed to the exec command. This allows the code in each file to be executed in the desired sequence and helps to automate the entire process.

### * * data to train model

The data required to train the nlp model that categorises the tweets can be found at:
https://drive.google.com/file/d/1xZhgOSXlPvt5ZumVaD47agfrX5LCYh3r/view?usp=share_link


### * * how to run the program

You need to apply for a twitter developer account and get access to get authentication keys and tweets via tweepy.  
Then run the following:
1. !git clone
2. cd thread_collector
3. pip install -r requirements.txt 
4. python3 main.py
