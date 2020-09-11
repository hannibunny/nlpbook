# Access Tweets

- Author:      Johannes Maucher
- Last update: 2020-09-09

This notebook demonstrates how the Python API [tweepy](http://www.tweepy.org/) can be applied to access data from twitter.

#!pip install tweepy

import tweepy
import json
from slugify import slugify

The tweepy authentification process is described in detail in [tweepy authentification](http://tweepy.readthedocs.io/en/v3.5.0/auth_tutorial.html). This process is implemented in the following two code-cells. It requires that corresponding personal twitter credentials are available. These credentials can be applied from [Twitter Apps](https://apps.twitter.com/).  

My personal credentials are saved in the file `twitterCredentials.json`. The contents of this file are loaded in the following code-cell.

with open("/Users/maucher/Keys/twitterCredentials.json") as data_file:    
    credentials = json.load(data_file)
#print credentials

Tweepy authentification:

auth = tweepy.OAuthHandler(credentials["consumerKey"], credentials["consumerSecret"])
auth.set_access_token(credentials["accessToken"],credentials["accessSecret"])
api = tweepy.API(auth)

The `API`-object can now be applied for example in order to read the own timeline of the Twitter homepage:

public_tweets = api.home_timeline(count=10)
for tweet in public_tweets:
    print("-"*10)
    print(tweet.author.name)
    print(tweet.text)

The `API`-object can also be applied for sending a new tweet (a.k.a. update status):

#api.update_status("This is just for testing tweepy api")

The API does not only provide access to content (tweets), but also to user-, friendship-, list-data and much more:

user = api.get_user('realDonaldTrump')
#ser = api.get_user('RegSprecher')
user.id

print(user.description)

user.followers_count

user.location

user.friends_count

for friend in user.friends():
    print(friend.screen_name)
print(len(user.friends()))

dirname="./TWEETS/"
users = ["mxlearn","reddit_python","ML_NLP","realDonaldTrump"]
numTweets=300
for user in users:
    print(user)
    user_timeline = api.user_timeline(screen_name=user, count=numTweets)
    filename = str(user) + ".json"
    with open(dirname+filename, 'w+',encoding="utf-8") as f:
        for idx, tweet in enumerate(user_timeline):
            tweet_text = user_timeline[idx].text
            #print(tweet_text)
            f.write(tweet_text + "\n")

Alternative way to access timeline using the cursor-object:

with open("twitterTimeline.json", 'w+') as f:
    for status in tweepy.Cursor(api.home_timeline).items(1):
        json.dump(status._json,f)

for friend in list(tweepy.Cursor(api.friends).items()):
    print(friend.name)

for tweet in tweepy.Cursor(api.user_timeline).items(1):
    print("-"*10)
    print(tweet.text)