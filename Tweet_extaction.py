import tweepy 
import re
import configparser
import pandas as pd

#Get Tweet data Export to excel
#Keyword = '#ดิวอริสรา since:2022-08-01 until:2022-08-31'
def gettweet(Keywords,limit):
    config = configparser.ConfigParser()
    config.read('config.ini')

    api_key = config['twitter']['api_key']
    api_key_secret = config['twitter']['api_key_secret']

    access_token = config['twitter']['access_token']
    access_token_secret = config['twitter']['access_token_secret']

    auth  = tweepy.OAuth1UserHandler(api_key,api_key_secret)
    auth.set_access_token(access_token,access_token_secret)
    api = tweepy.API(auth)
    Keyword = Keywords +'exclude:retweets'
    tweets = tweepy.Cursor(api.search_tweets, q=Keyword,count=100,tweet_mode='extended').items(limit)

    fields = ['created_at', 'Text',
            'retweet_count',
            'favorite_count']

    tweets_list = [[tweet.created_at, 
                    tweet.full_text,
                    tweet.retweet_count,
                    tweet.favorite_count] for tweet in tweets]

    data = pd.DataFrame(data= tweets_list,columns = fields)
#data = pd.DataFrame(data=[tweet.full_text for tweet in tweets],columns=['Text'])
    data['year'] = pd.DatetimeIndex(data['created_at']).year
    data['month'] = pd.DatetimeIndex(data['created_at']).month
    data['days'] = pd.DatetimeIndex(data['created_at']).day

#Remove RT and Spacial Character

    remove_rt = lambda x: re.sub('RT @\w+: '," ",x)
    #remove_Hastage = lambda x: re.sub(r'#', '',x)
    sp_char = lambda x :re.sub('([^a-zA-Z^\u0E00-\u0E7F])|(#)','',x)
    https_re = lambda x :re.sub(r'http\S+','',x)
    data["Text"] = data.Text.map(remove_rt)
    data["Text"] = data.Text.map(https_re).map(sp_char)
    #data["Text"] = data.Text.map(remove_Hastage)
    data["Text"] = data.Text.str.lower()
    data2 = data.drop_duplicates(subset=['Text'],ignore_index= True).reset_index(drop=True)
    data2['Text'].to_excel("test_output.xlsx") 
gettweet('#สนใจแอดไลน์',50)