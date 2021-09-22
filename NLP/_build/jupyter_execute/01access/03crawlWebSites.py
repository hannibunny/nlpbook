#!/usr/bin/env python
# coding: utf-8

# # Download HTML Files
# 
# - Author:      Johannes Maucher
# - Last update: 2018-10-12
# 
# 

# In[1]:


#!pip install slugify


# In[2]:


import bs4
import requests
from slugify import slugify
import datetime
import os
from urllib.request import urlopen


# ## Define lists of news domains for different languages and categories

# In[3]:


today=datetime.datetime.now().strftime("%Y-%m-%d")
print(today)


# In[4]:


cat="GENERAL"
#cat="TECH"
#cat="SPORT"
lang="GERMAN"
#lang="ENGLISH"


# In[5]:


general_sources_de = ['http://www.zeit.de',
                      'http://www.spiegel.de/',
                      'http://www.welt.de/',
                      'http://www.sueddeutsche.de',
                      'http://www.faz.net'
                     ]

general_sources_en = ['https://www.washingtonpost.com',
                      'http://www.nytimes.com/',
                      'http://www.chicagotribune.com/',
                      'http://www.bostonherald.com/',
                      'http://www.sfchronicle.com/']

tech_sources_de=['http://chip.de/',
                 'http://t-online.de',
                 'http://www.computerbild.de',
                 'http://www.heise.de',
                 'http://www.golem.de']

tech_sources_en=['http://radar.oreilly.com/',
                 'https://www.cnet.com/news/',
                 'http://www.techradar.com/news/computing'
                ]                 


# In[6]:


if lang=="ENGLISH":
    if cat=="GENERAL":
        sources=general_sources_en
    else:
        sources=tech_sources_en
else:
    if cat=="GENERAL":
        sources=general_sources_de
    else:
        sources=tech_sources_de
    
    


# ## Download subdomain HTML pages 
# The below defined function `crawl()` determines all subdomains of the specified url and saves the HTML files of these subdomains.

# In[7]:


def crawl(url,maxSubSites=5,category="GENERAL",lang="GERMAN"):
    domain = url.split("//www.")[-1].split("/")[0]
    print(domain)
    dirname=lang+"/"+category+"/"+"HTML/"+domain.split('.')[0]+"-"+today
    try:
        os.makedirs(dirname)
    except:
        print("Directory %s already exists."%dirname)
    filename = dirname+"/"+domain.split('.')[0] + '.html'
    html = requests.get(url).content
    with open(filename, 'wb') as f:
        #print filename
        f.write(html)
    soup = bs4.BeautifulSoup(html, "html.parser")
    links = set(soup.find_all('a', href=True)) #find all links in this page
    count=0
    for link in links:
        if count > maxSubSites:
            break
        sub_url = link['href']
        page_name = link.string
        if domain in sub_url:
            count+=1
            try:
                page = requests.get(sub_url).content
                filename = dirname+"/"+slugify(page_name).lower() + '.html'
                with open(filename, 'wb') as f:
                    #print filename
                    f.write(page)
            except:
                pass
    return dirname


# In[8]:


htmlDirs=[]
for url in sources:
    htmldir=crawl(url,maxSubSites=50,category=cat,lang=lang)
    htmlDirs.append(htmldir)


# In[10]:


print(htmlDirs)


# ## Crawl raw text from local HTML files

# In[11]:


TAGS = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'p']
def read_html(path,minlen=15):
    with open(path, 'r') as f:
        html = f.read()
        soup = bs4.BeautifulSoup(html, "html.parser")
        for tag in soup.find_all(TAGS):
            text = tag.get_text()
            if len(text)>minlen:
                yield text


# In[12]:


rawtextlist=[]
for dir in htmlDirs:
    textdir=dir.replace("HTML","TEXT")
    try:
        os.makedirs(textdir)
    except:
        pass
    for htmlfile in os.listdir(dir):
        htmlpath=dir+"/"+htmlfile
        rawText=read_html(htmlpath)
        rawtextlist.append(rawText)
        textpath=htmlpath.replace("HTML","TEXT")
        textfilename=textpath.replace(".html",".txt")
        with open(textfilename,"w") as f:
            for cont in rawText:
                f.write(cont)
        f.close()
#print len(rawtextlist)
        


# ## Questions
# 1. Inspect some of the created raw-text files and suggest improvements.
