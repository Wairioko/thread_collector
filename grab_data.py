import os
import time
# from datetime import datetime
import tweepy
import pandas as pd

API_KEY = 'WHqCviEwyimfpPBFUdbimIs8V'
API_KEY_SECRET = '6NbaVOSnvKqkx94uAZks0ZrTLdiD8UC0v03YlMCutkNi1sOg0t'
ACCESS_TOKEN = '1574326318241554434-XPneVnJpNOvF2PA3SPOSQmp8I3LefM'
ACCESS_TOKEN_SECRET = '3ALrjPbaqhoGbS4IBInKuDuN5UYHjnNxz0pwl8XaA3i2U'
BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAKAChgEAAAAACr63I1jRBhLTgSwNHYu9P' \
               'LkuZFk%3D47ZRV9j0nMjzSQHA9nKjmQ5YFXQrC2qwoNdHFcs30rtos' \
               'llE3C'

auth = tweepy.OAuthHandler(API_KEY, API_KEY_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True)
client = tweepy.Client(bearer_token=BEARER_TOKEN, consumer_key=API_KEY, consumer_secret=API_KEY_SECRET,
                       access_token=ACCESS_TOKEN, access_token_secret=ACCESS_TOKEN_SECRET
                       )
# streaming_client = tweepy.StreamingClient(BEARER_TOKEN)


query = 'thread OR 🧵 '
# csvFile = open("tweets.csv", 'a')


def scalptweets():
    db_tweets = pd.DataFrame(columns=['user', 'tweet_id', 'retweetcount', 'text'])
    program_start = time.time()

    for i in range(0, 6):
        start_run = time.time()
        tweets = tweepy.Cursor(api.search_tweets, q=query, lang="en", tweet_mode='extended').items(2500)
        tweet_list = [tweet for tweet in tweets]
        notweets = 2500

        for tweet in tweet_list:
            retweetcount = tweet.retweet_count
            user_name = tweet.user.screen_name
            tweet_id = tweet.id
            try:
                text = tweet.retweeted_status.full_text
            except AttributeError:
                text = tweet.full_text

            ith_tweet = [user_name, tweet_id, retweetcount, text]
            db_tweets.loc[len(db_tweets)] = ith_tweet
            notweets += 1
        end_run = time.time()
        duration_run = round((end_run-start_run)/60, 2)

        print('no. of tweets scraped for run {} is {}'.format(i + 1, notweets))
        print('time take for {} run to complete is {} mins'.format(i+1, duration_run))
        time.sleep(920)
        # Obtain timestamp in a readable format
        # to_csv_timestamp = datetime.today().strftime('%Y%m%d_%H%M%S')
        # + to_csv_timestamp
        path = os.getcwd()
        filename = path + 'tweets.csv'
        db_tweets.to_csv(filename, index=False)
        program_end = time.time()
        print('Scraping has completed!')
        print('Total time taken to scrap is {} minutes.'.format(round(program_end - program_start)/60, 2))


scalptweets()
