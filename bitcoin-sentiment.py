# import tweepy library for twitter api access and textblob libary for sentiment analysis
import csv
import tweepy
import numpy as np
from textblob import TextBlob

# set twitter api credentials
consumer_key= 'TWITTER_CONSUMER_KEY'
consumer_secret= 'TWITTER_CONSUMER_SECRET'
access_token='TWITTER_ACCESS_TOKEN'
access_token_secret='TWITTER_ACCESS_TOKEN_SECRET'

# access twitter api via tweepy methods
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
twitter_api = tweepy.API(auth)

# fetch tweets by keywords
tweets = twitter_api.search(q=['bitcoin, price'], count=100)


def get_polarity(tweets):
    # run polarity analysis on tweets

    tweet_polarity = []

    for tweet in tweets:
        analysis = TextBlob(tweet.text)
        tweet_polarity.append(analysis.sentiment.polarity)

    return tweet_polarity


def get_subjectivity(tweets):
    # run subjectivity analysis on tweets

    tweet_subjectivity = []

    for tweet in tweets:
        analysis = TextBlob(tweet.text)
        tweet_subjectivity.append(analysis.sentiment.subjectivity)

    return tweet_subjectivity


def classify_sentiment(analysis, threshold = 0):
    # classify sentiment polarity as positive or negative

    if analysis.sentiment.polarity > threshold:
        return 'Positive'
    elif analysis.sentiment.polarity < threshold:
        return 'Negative'
    else:
        return 'Neutral'


# get polarity and subjectivity data
polarity = get_polarity(tweets)
subjectivity = get_subjectivity(tweets)

# print sentiment stats
print('Polarity count: %s' % np.count_nonzero(polarity))
print('Polarity average: %.3f' % np.mean(polarity))
print('Polarity standard deviation: %.3f' % np.std(polarity))
print('Polarity coefficient of variation: %.3f' % (np.std(polarity) / np.mean(polarity)))
print('********')
print('Subjectivity count: %s' % np.count_nonzero(subjectivity))
print('Subjectivity average: %.3f' % np.mean(subjectivity))
print('Subjectivity standard deviation: %.3f' % np.std(subjectivity))
print('Subjectivity coefficient of variation: %.3f' % (np.std(subjectivity) / np.mean(subjectivity)))

# save tweets, polarity, subjectivity, and sentiment class to csv file
path = 'CSV_FILE_PATH'

with open(path, 'w') as f:
    writer = csv.writer(f)
    f.write('tweet, polarity, subjectivity, sentiment_class\n')

    for tweet in tweets:
        analysis = TextBlob(tweet.text)
        writer.writerow([tweet.text.encode('utf8'), analysis.sentiment.polarity, analysis.sentiment.subjectivity, classify_sentiment(analysis)])

f.close()
