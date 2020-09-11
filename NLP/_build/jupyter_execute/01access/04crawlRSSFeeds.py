# Access RSS Feed

- Author:      Johannes Maucher
- Last update: 2020-09-09

This notebook demonstrates how the contents of an RSS feed can be extracted and saved to a file. For each of the parsed feeds
1. the short-texts of all current posts are stored in a single file. The file-name is the time `hh:mm` when the feed was parsed. I.e for each new parsing of a feed a new file with the short-texts of all current posts is allocated. The name of the folder, which contains these files, contains the current date and the name of the feed.
2. for each post the .html page, which contains the full message is stored (optionally).
3. for each post the raw-text of the .html page, which contains the full message is stored (optionally.)

## Feedparser
The contents of a RSS feeds can easily parsed by the python [feedparser](https://pypi.python.org/pypi/feedparser) package. This package must be installed and imported. The `parse()`-method of the feedparser returns the contents of the feed in a structured form. This return-object usually consists of a set of *entries*, which represent the current messages in the feed. The relevant text of an entry is contained in the fields `title` and `description`. Moreover, each entry has a `link`-element, which refers to the .html site of the full article.

import feedparser
import requests
import bs4
import datetime
import os

## Define category and language of feeds
For each of the supported languages (currently *German* and *English*) and each of the supported categories (currently *general* and *tech*), a list of feeds is provided. The directories for storing the parsed contents, are sturured with respect to the selected language and category.
* The name of the top-level directory is the language name, i.e. currently either `GERMAN` or `ENGLISH`,
* The name of the 2nd level directory is the category name, i.e. currently either `GENERAL` or `TECH`,
* In the 3rd level currently there is only one folder `RSS`.
* In the 4th level, 3 directories exist:
    * `FeedText`
    * `FullText`
    * `HTML`
* 5th level: Each of the 3 directories in the 4th level has an arbitrary number of subdirectories, each identified by `FEEDNAME-DATE`. 
* 6th level under 
    * `FeedText`: for each parsing process a single file, whose name is the time of parsing and whose contents are all short-texts in this feed at the time of parsing.
    * `HTML`: All .html files of the full messages, to which the links in the feed-posts point to.
    * `FullText`: The parsed raw text of all .html files of the full messages, to which the links in the feed-posts point to.

Choose the language and category:

today=datetime.datetime.now().strftime("%Y-%m-%d")
#cat="GENERAL"
cat="TECH"
#cat="SPORT"
#lang="GERMAN"
lang="ENGLISH"

Some feed lists for all languages and categories:

tech_feeds_en = [{'title': 'districtdatalabs', 'link': 'http://blog.districtdatalabs.com/feed'},
                 {'title': 'oreillyradar', 'link': 'http://feeds.feedburner.com/oreilly/radar/atom'},
                 {'title': 'kaggle', 'link': 'http://blog.kaggle.com/feed/'},
                 {'title': 'revolutionanalytics', 'link': 'http://blog.revolutionanalytics.com/atom.xml'}]
general_feeds_en = [{'title': 'bbcnews', 'link': 'http://feeds.bbci.co.uk/news/rss.xml?edition=uk'},
                 {'title': 'bbcbusiness', 'link': 'http://feeds.bbci.co.uk/news/business/rss.xml?edition=uk'},
                 {'title': 'bbcpolitics', 'link': 'http://feeds.bbci.co.uk/news/politics/rss.xml?edition=uk'},
                 {'title': 'washingtonpostworld', 'link': 'http://feeds.washingtonpost.com/rss/world'},
                 {'title': 'washingtonpostpolitics', 'link': 'http://feeds.washingtonpost.com/rss/politics'}]
tech_feeds_de = [{'title': 'tonline', 'link': 'http://rss1.t-online.de/c/11/53/06/84/11530684.xml'},
                 {'title': 'computerbild', 'link': 'http://www.computerbild.de/rssfeed_2261.xml?node=13'},
                 {'title': 'heise', 'link': 'http://www.heise.de/newsticker/heise-top-atom.xml'},
                 {'title': 'golem', 'link': 'http://rss.golem.de/rss.php?r=sw&feed=RSS0.91'},
                 {'title': 'chip', 'link': 'http://www.chip.de/rss/rss_topnews.xml'}]
general_feeds_de = [{'title': 'zeit', 'link': 'http://newsfeed.zeit.de/index'},
                    {'title': 'zeit_wirtschaft', 'link': 'http://newsfeed.zeit.de/wirtschaft/index'},
                    {'title': 'zeit_politik', 'link': 'http://newsfeed.zeit.de/politik/index'},
                    {'title': 'welt_wirtschaft', 'link': 'http://www.welt.de/wirtschaft/?service=Rss'},
                    {'title': 'welt_politik', 'link': 'http://www.welt.de/politik/?service=Rss'},
                    {'title': 'spiegel', 'link': 'http://www.spiegel.de/schlagzeilen/tops/index.rss'},
                    {'title': 'sueddeutsche', 'link': 'http://www.sueddeutsche.de/app/service/rss/alles/rss.xml'},
                    {'title': 'faz', 'link': 'http://www.faz.net/rss/aktuell/'}
                    ]

For the defined language and category, determine a suitable feedlist:

if lang=="ENGLISH":
    if cat=="GENERAL":
        feeds=general_feeds_en
    else:
        feeds=tech_feeds_en
else:
    if cat=="GENERAL":
        feeds=general_feeds_de
    else:
        feeds=tech_feeds_de

TAGS = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'p']
MINLEN=30 #Write only texts, whose length (number of characters) exceeds this size.

def rss_parse(feed,category="GENERAL",lang="GERMAN",savehtml=False,savefulltext=True):
    #Create directories
    dirname=lang+"/"+category+"/"+"RSS/FeedText/"+feed["title"]+"-"+today+"/"
    htmldirname=lang+"/"+category+"/"+"RSS/HTML/"+feed["title"]+"-"+today+"/"
    fulldirname=lang+"/"+category+"/"+"RSS/FullText/"+feed["title"]+"-"+today+"/"
    try:
        os.makedirs(dirname)
        print("Created: ",dirname)
    except:
        print("Directory %s already exists."%dirname)
    try:
        os.makedirs(htmldirname)
    except:
        print("Directory %s already exists."%htmldirname)
    try:
        os.makedirs(fulldirname)
    except:
        print("Directory %s already exists."%fulldirname)
    
    # Parse feed
    parsed = feedparser.parse(feed["link"])
    posts = parsed.entries
    now=datetime.datetime.now().strftime("%H-%M")
    cummulativeFile=now+ '.txt'
    feedfilename = dirname+cummulativeFile
    for post in posts:
        # Get and save short-texts of feed-posts (all texts of one feed at one time in a common file)
        try:
            soup = bs4.BeautifulSoup(post.title+"\n"+post.description, "lxml")
        except:
            continue

        text = soup.get_text()
        post_title = post.title
        with open(feedfilename, 'a',encoding="utf-8") as f:
            text=text+ "\n \n"
            try:
                f.write(text)
            except:
                print("CAN'T WRITE FEEDTEXT TO FILE")
                print(text)
        
        # Go to the .html-site of the full message
        page = requests.get(post.link).content
        if savehtml:
            htmlfilename = htmldirname+post.title.lower() + '.html'
            with open(htmlfilename, 'w',encoding="utf-8") as f:
                f.write(page.decode("utf-8"))
            f.close()
        
        # Parse raw text from the .html page of the full message
        if savefulltext:
            fullfilename = fulldirname+post.title.lower() + '.txt'
            soup = bs4.BeautifulSoup(page, "lxml")
            with open(fullfilename, 'w',encoding="utf-8") as ft:
                for tag in soup.find_all(TAGS):
                    text = tag.get_text()
                    if len(text)>MINLEN:
                        ft.write(text)
            f.close()

for f in feeds:
    rss_parse(f,category=cat,lang=lang,savehtml=True,savefulltext=True)

