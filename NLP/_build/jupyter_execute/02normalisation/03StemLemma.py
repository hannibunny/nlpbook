#!/usr/bin/env python
# coding: utf-8

# TextBlob Stemming and Lemmatization
# =====================================
# 
# - Author:      Johannes Maucher
# - Last update: 2021-10-26
# 
# In this notebook the application of stemming and lemmatization shall be demonstrated. For this we apply the correponding modules provided by the NLP package [**TextBlob**](https://textblob.readthedocs.io/en/dev/). Since stemming and lemmatization both require the segmentation of texts into lists of words, segmentation and other preprocessing-functions of TextBlob are also shown. In notebook [Regular Expressions](../01access/05RegularExpressions.ipynb) it has already been demonstrated how to implement segmentation in Python without additional packages. If you like to go directly to [Word Normalisation click here](#word_normalisation). 
# 
# [**TextBlob**](https://textblob.readthedocs.io/en/dev/) is a Python library for Natural Language Processing. It provides a simple API for, e.g.
# * Noun phrase extraction
# * Part-of-speech tagging
# * Sentiment analysis
# * Classification
# * Language translation and detection powered by Google Translate
# * Tokenization (splitting text into words and sentences)
# * Word and phrase frequencies
# * Parsing
# * n-grams
# * Word inflection (pluralization and singularization) and lemmatization
# * Spelling correction
# * WordNet integration

# In[1]:


#!pip install textblob


# In[2]:


import textblob
print(textblob.__version__)


# In[3]:


from textblob import TextBlob


# ## TextBlob objects
# TextBlob objects are like Python strings, which have enhanced with typical NLP processing methods. They are generared as follows:

# In[4]:


myBlob1=TextBlob("""TextBlob is a Python (2 and 3) library for processing textual data. 
It provides a simple API for diving into common natural language processing (NLP) tasks 
such as part-of-speech tagging, noun phrase extraction, sentiment analysis, classification, 
translation, and more. New York is a nice city.""")
myBlob2=TextBlob(u"Das Haus steht im Tal auf der anderen Seite des Berges. Es gehört Dr. med. Brinkmann und wird von der Immo GmbH aus verwaltet. Im Haus befindet sich u.a. eine Physio-Praxis.")


# ## Language Detection

# In[5]:


#print("Blob1 text language is ",myBlob1.detect_language())
#print("Blob2 text language is ",myBlob2.detect_language())


# ## Sentences and Words
# The TextBlob class also integrates methods for tokenisation. The corresponding segmentation of the given text into sentences and words can be obtained as follows:
# 
# **English text:**

# In[6]:


for s in myBlob1.sentences:
    print("-"*20)
    print(s)


# In[7]:


myBlob1.sentences[2].words


# As can be seen in the example above multi-word expressions, like *New York* are not detected.

# **German text:**

# In[8]:


for s in myBlob2.sentences:
    print("-"*20)
    print(s)


# As can be seen in the example above, segmentation works fine for English texts, but fails for German texts, in particular if the text contains abbreviations, which are not used in English. However, the [**TextBlob-DE package**](https://pypi.python.org/pypi/textblob-de) provides German language support for *TextBlob*.

# ## Part-of-Speech Tagging

# [Meaning of POS Tags according to Penn Treebank II tagset](https://gist.github.com/nlothian/9240750).

# In[9]:


for word,pos in myBlob1.sentences[0].tags:
    print(word, pos)


# Doesn't work for German:

# In[10]:


for word,pos in myBlob2.sentences[0].tags:
    print(word, pos)


# <a id='word_normalisation'></a>
# ## Word Normalization
# In NLP the term *word normalization* comprises methods to map different word forms to a unique form. Word normalization reduces complexity and often improves the NLP task. E.g. in document-classification it usually does not matter in which temporal form a verb is written. The mapping of all temporal forms to the base form reduces the number of words in the vocabulary and likely increases the accuracy of the classifier.
# 
# ### Singularization
# One form of word normalization is to map all words in plural into the corresponding singular form. With *textblob* this can be realized as follows:

# In[11]:


myBlob6=TextBlob("The cars in the streets around Main Square have been observed by policemen")
for word in myBlob6.words:
    print(word, word.singularize())


# ### Lemmatization
# Lemmatization maps all distinct forms of a word to the baseform. E.g. the wordforms `went, gone, going` are all mapped to the baseform `go`:

# In[12]:


print(TextBlob("went").words[0].lemmatize("v")) # mode "v" for verb
print(TextBlob("gone").words[0].lemmatize("v"))
print(TextBlob("going").words[0].lemmatize("v"))


# Lemmatization of adverbs and adjectives:

# In[13]:


print(TextBlob("later").words[0].lemmatize("a")) # mode "a" for adverb/adjective
print(TextBlob("prettier").words[0].lemmatize("a"))
print(TextBlob("worse").words[0].lemmatize("a"))


# Lemmatizsation of nouns: 

# In[14]:


print(TextBlob("women").words[0].lemmatize("n")) # mode "n" for noun


# In[15]:


myBlob3=TextBlob("The engineer went into the garden and found the cats lying beneath the trees")


# In[16]:


for word in myBlob3.words:
    print(word, word.lemmatize("v"))


# ### Stemming
# The drawback of lemmatization is its complexity. Some languages have a quite regular structure. This means that the different word-forms can be derived from the baseform by a well defined set of rules. In this case the inverse application of these rules can be applied for determining the baseform. However, languages such as German, have many irregular cases. Lemmatization then requires a *dictionary*, which lists all different word-forms an their corresponding baseform.
# 
# Stemming is simple and less complex alternative to lemmatization. Stemmers map each word to their stem. In contrast to lemmatization the result of stemming need not be a lexical entry (valid word). Stemmers, e.g. the *Porter Stemmer* apply heuristics for word-suffixes and strip-off found suffixes from the word. E.g. in the word `engineer` a stemmer finds `er` as a frequnet suffix. It strips-off this suffix and outputs the found stem `engin`.  

# In[17]:


for word in myBlob3.words:
    print(word, word.stem())


# ### Word correction
# Word correction can also be considered as type of normalization. It maps misspelled words to likely correct form. 

# In[18]:


w=TextBlob("sentense is not tru")
wcorr=w.correct()
wcorr


# ## TextBlob-DE for German Language

# In[19]:


#!pip install textblob-de


# In[20]:


import textblob_de


# In[21]:


myBlob4=textblob_de.TextBlobDE(u"Das Haus steht im Tal auf der anderen Seite des Berges. Es gehört Dr. med. Brinkmann und wird von der Immo GmbH verwaltet. Im Haus befindet sich u.a. eine Physio-Praxis.")


# ### Sentences and Words

# In[22]:


for s in myBlob4.sentences:
    print("-"*20)
    print(s)


# In[23]:


myBlob4.sentences[1].words


# In[24]:


myBlob4.sentences[1].noun_phrases


# ### Part-of-Speech Tagging

# In[25]:


myBlob5=textblob_de.TextBlobDE("Er ist mit seinen Katzen über drei Tische gesprungen")


# In[26]:


for word,tag in myBlob5.tags:
    print(word, tag)


# ### Lemmatization

# In[27]:


for word in myBlob5.words.lemmatize():
    print(word)


# In[ ]:




