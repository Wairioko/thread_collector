import pandas as pd
import os
from furl import furl
path = os.getcwd()
categorised_tweets = pd.read_csv(path + 'new_clean_tweets.csv')
links = []
for index, row in categorised_tweets.iterrows():
    link = furl("https://twitter.com/{0}/status/{1}".format(row['user'], row['tweet_id']))
    links.append(link)

categorised_tweets['links'] = links
urled_tweets = categorised_tweets.reindex(columns=['links', 'category', 'text', 'retweetcount', 'user', 'tweet_id'])
filename = path + 'weekly_tweets.csv'
urled_tweets.to_csv(filename)
