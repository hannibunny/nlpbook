# Access Contents of HTML Page

- Author:      Johannes Maucher
- Last update: 2018-10-21

This notebook demonstrates how to parse a HTML document and access dedicated elements of the parse tree.
[Beautiful Soup](http://www.crummy.com/software/BeautifulSoup/bs4/doc/#) is a python package for parsing HTML. Download and install version 4 by typing:

> `pip install beautifulsoup4`

into the command shell. Once it is installed it can be imported by

from bs4 import BeautifulSoup

For accessing arbitrary resources by URL the python modul [urllib](https://docs.python.org/2/library/urllib.html) must also be installed. Import the method _urlopen()_ from this module:  

from urllib.request import urlopen

If these two modules are available the HTML parse tree of the specified URL can easily be generated as follows.

#url="http://www.zeit.de"
url="http://www.spiegel.de"
#url="http://www.sueddeutsche.de"
html=urlopen(url).read()
soup=BeautifulSoup(html,"html.parser")

Now e.g. the title of the URL can be accessed by:

titleTag = soup.html.head.title
print("Title of page:  ",titleTag.string)

## Get all links in the page
All links in the page can be retrieven by the following code (only the first 20 links are printed)

hreflinks=[]
Alllinks=soup.findAll('a') #The <a> tag defines a hyperlink, which is used to link from one page to another.
for l in Alllinks:
    if l.has_attr('href'):
        hreflinks.append(l)
print("Number of links in this page: ",len(hreflinks))
for l in hreflinks[:20]:
    print(l['href'])

## Get all news titles
Get title of all news, which are currently listed on [www.zeit.de](http://www.zeit.de):

#print soup.get_text()hreflinks=[]
AllTitles=soup.findAll('h2')
alltitles=[]
alltitleLinks=[]
for l in AllTitles:
    #print l
    try:
        title = l.find('a')['title']
        link = l.find('a')['href']
        print('-'*40)
        print(title)
        print(link)
        alltitles.append(title)
        alltitleLinks.append(link)
    except:
        pass

## Get all images of the page

Get url of all images, which are currently displayed on [www.zeit.de](http://www.zeit.de):

imglinks=[]
AllImgs=soup.findAll('img')
for l in AllImgs:
    if l.has_attr('src'):
       imglinks.append(l)

for l in imglinks[:10]:
    print(l['src'])

## Get entire text of a news-article

IDX=0
suburl=alltitleLinks[IDX]
try:
    html=urlopen(suburl).read() #works if subdomains are referenced by absolute path
except:
    html=urlopen(url+suburl).read() #works if subdomains are referenced by relative path
soup=BeautifulSoup(html,"html.parser")
AllP=soup.findAll('p')
for p in AllP:
    print(p.get_text())

## Questions and Remarks
1. This notebook demonstrates how raw-text can be crawled from news-sites. But what is the drawback of this method?
2. Execute the entire notebook also for `www.spiegel.de` and `www.sueddeutsche.de`.
3. What do you observe? How to solve the problem?

