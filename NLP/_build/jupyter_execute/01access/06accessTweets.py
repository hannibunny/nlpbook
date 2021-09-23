#!/usr/bin/env python
# coding: utf-8

# # Access Tweets
# 
# - Author:      Johannes Maucher
# - Last update: 2020-09-09
# 
# This notebook demonstrates how the Python API [tweepy](http://www.tweepy.org/) can be applied to access data from twitter.

# In[1]:


#!pip install tweepy


# In[2]:


import tweepy
import json
from slugify import slugify


# The tweepy authentification process is described in detail in [tweepy authentification](http://tweepy.readthedocs.io/en/v3.5.0/auth_tutorial.html). This process is implemented in the following two code-cells. It requires that corresponding personal twitter credentials are available. These credentials can be applied from [Twitter Apps](https://apps.twitter.com/).  
# 
# My personal credentials are saved in the file `twitterCredentials.json`. The contents of this file are loaded in the following code-cell.

# In[3]:


with open("/Users/johannes/Dropbox (Privat)/twitterCredentials.json") as data_file:    
    credentials = json.load(data_file)
#print credentials


# Tweepy authentification:

# In[4]:


auth = tweepy.OAuthHandler(credentials["consumerKey"], credentials["consumerSecret"])
auth.set_access_token(credentials["accessToken"],credentials["accessSecret"])
api = tweepy.API(auth)


# The `API`-object can now be applied for example in order to read the own timeline of the Twitter homepage:

# In[5]:


public_tweets = api.home_timeline(count=10)
for tweet in public_tweets:
    print("-"*10)
    print(tweet.author.name)
    print(tweet.text)


# The `API`-object can also be applied for sending a new tweet (a.k.a. update status):

# In[6]:


#api.update_status("This is just for testing tweepy api")


# The API does not only provide access to content (tweets), but also to user-, friendship-, list-data and much more:

# In[7]:


#user = api.get_user('realDonaldTrump')
user = api.get_user('RegSprecher')
user.id


# In[8]:


print(user.description)


# In[9]:


user.followers_count


# In[10]:


user.location


# In[11]:


user.friends_count


# In[12]:


for friend in user.friends():
    print(friend.screen_name)
print(len(user.friends()))


# In[13]:


dirname="./TWEETS/"
users = ["mxlearn","reddit_python","ML_NLP"]
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


# Alternative way to access timeline using the cursor-object:

# In[14]:


with open("twitterTimeline.json", 'w+') as f:
    for status in tweepy.Cursor(api.home_timeline).items(1):
        json.dump(status._json,f)


# In[15]:


for friend in list(tweepy.Cursor(api.friends).items()):
    print(friend.name)


# In[16]:


for tweet in tweepy.Cursor(api.user_timeline).items(1):
    print("-"*10)
    print(tweet.text)


# In[ ]:




