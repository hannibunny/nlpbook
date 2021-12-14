#!/usr/bin/env python
# coding: utf-8

# # Implementation of Topic Extraction and Document Clustering
# 
# - Author:      Johannes Maucher
# - Last update: 14.12.2021
# 
# This notebook demonstrates how [gensim](http://radimrehurek.com/gensim/) can be applied for *Latent Semantic Indexing (LSI)*. In LSI a set of abstract topics (features), which are latent in a set of simple texts, is calculated. Then the documents are described and visualised with respect to these abstract features. The notebook is an adoption of the corresponding [gensim LSI tutorial](http://radimrehurek.com/gensim/tut2.html). 

# ## Collect and filter text documents
# A list of very small documents is defined. From the corresponding BoW (Bag of Words) representation all stopwords and all words, which appear only once are removed. The resulting cleaned BoW models of all documents are printed below.  

# In[1]:


#!pip install --upgrade gensim


# In[2]:


from gensim import corpora, models, similarities

documents = ["Human machine interface for lab abc computer applications",
              "A survey of user opinion of computer system response time",
              "The EPS user interface management system",
              "System and human system engineering testing of EPS",
              "Relation of user perceived response time to error measurement",
              "The generation of random binary unordered trees",
              "The intersection graph of paths in trees",
              "Graph minors IV Widths of trees and well quasi ordering",
              "Graph minors A survey"]
# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]
# remove words that appear only once
all_tokens=[]
for t in texts:
    for w in t:
        all_tokens.append(w)
tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
texts = [[word for word in text if word not in tokens_once]
          for text in texts]
for t in texts:
    print(t)


# ## Dictionaries and Corpora
# The words of the cleaned documents constitute a dictionary, which is persistently saved in the file *deerwester.dict*. The dictionary-method *token2id* displays the dictionary indes of each word.

# In[3]:


dictionary = corpora.Dictionary(texts)
dictionary.save('deerwester.dict') # store the dictionary, for future reference
print(dictionary.token2id)


# Next, a corpus is generated, which is a very efficient representation of the cleaned documents. In the corpus each word is represented by it's index in the dictionary. The corpus is persistently saved to file *deerwester.mm*.

# In[4]:


corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('deerwester.mm', corpus) # store to disk, for later use


# In[5]:


for c in corpus:
    print(c)


# The following code snippet demonstrates how a dictionary and a corpus can be loaded into the python program.

# In[6]:


dictionary = corpora.Dictionary.load('deerwester.dict')
corpus = corpora.MmCorpus('deerwester.mm')
for c in corpus:
    print(c)


# ## TF-IDF Model of the corpus
# A tf-idf model is generated from the cleaned documents of the corpus and all corpus documents are represented by the vector of tf-idf values of their words.

# ### TF-IDF Model without document-vector normalisation

# In[7]:


tfidf = models.TfidfModel(corpus,normalize=False)  # generate a transformation object and fit it to the corpus documents
corpus_tfidf = tfidf[corpus] # apply the transformation to all corpus documents
for doc in corpus_tfidf:
    print(doc)


# Transform a new document to tf-idf vector. The new document in this example consists of the words 
# * *computer (index 0)*, 
# * *human (index 1)* 
# * 2 times the word *system (index 5)*:

# In[8]:


newDoc=[(0,1),(1,1),(5,2)]
newTFIDF=tfidf[newDoc]
print(newTFIDF)


# Verify that log2 is applied in the tf-idf calculation:

# In[9]:


import numpy as np
np.log2(9/2)


# ### TF-IDF Model with document-vector normalisation

# In general it is better to normalise the document vectors, such that each vector has a length of $1$. By applying document normalisation the obtained vectors are *independent* of document length.

# In[10]:


tfidf = models.TfidfModel(corpus,normalize=True)  # generate a transformation object and fit it to the corpus documents
corpus_tfidf = tfidf[corpus] # apply the transformation to all corpus documents
for doc in corpus_tfidf:
    print(doc)


# ## LSI Model of the corpus
# A Latent Semantic Indexing (LSI) model is generated from the given documents. The number of topics that shall be extracted is selected to be two in this example:

# In[11]:


lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2) # initialize an LSI transformation
corpus_lsi = lsi[corpus_tfidf]
lsi.print_topics(2)


# As shown below, each document is described in the new 2-dimensional space. The dimensions represent the two extracted topics.

# In[12]:


x=[]
y=[]
i=0
for doc in corpus_lsi: # both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly
    print("Document %2d: \t"%i,doc)
    x.append(doc[0][1])
    y.append(doc[1][1])
    i+=1


# The documents can be plotted in the new 2-dimensional space. In this space the documents are clearly partitioned into 2 clusters, each representing one of the 2 topics.

# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
plt.figure(figsize=(12,10))
plt.plot(x,y,'or')
plt.title('documents in the new space')
plt.xlabel('topic 1')
plt.ylabel('topic 2')
#plt.xlim([0,1.1])
#plt.ylim([-0.9,0.3])
s=0.02
for i in range(len(x)):
    plt.text(x[i]+s,y[i]+s,"doc "+str(i))
plt.show()


# LSI models can be saved to and loaded from files: 

# In[14]:


lsi.save('model.lsi') # same for tfidf, lda, ...
lsi = models.LsiModel.load('model.lsi')

