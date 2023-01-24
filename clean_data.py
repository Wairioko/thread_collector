from furl import furl
import os
import pandas as pd

# read data from csv file
path = os.getcwd()
tweets_data = pd.read_csv(path + '/tweets.csv')


def clean_tweets(data):
    # find tweets with a retweet count above 700 and containing either thread or 🧵 in text
    new_tweets = (data.query('retweetcount.values >= 700 & text.str.contains("thread").values '
                                    ' | text.str.contains("🧵").values'))
    # sort the result by size retweet count
    arrange_tweets = new_tweets.sort_values(by=['retweetcount'], ascending=False)
    # dropping ALL duplicate tweets but keeping first(with largest engagement)
    drop_duplicates = arrange_tweets.drop_duplicates(subset=['text'], keep='first')
    # creating new file with new set of tweets
    filename = path + 'clean_tweets.csv'
    return drop_duplicates.to_csv(filename, index=False)


clean_tweets(tweets_data)
