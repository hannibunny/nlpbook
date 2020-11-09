# Topic Extraction in RSS-Feed Corpus

- Author:      Johannes Maucher
- Last update: 2018-11-16

In the [notebook 01gensimDocModelSimple](01gensimDocModelSimple.ipynb) the concepts of dictionaries, document models, tf-idf and similarity have been described using an example of a samll document collection. Moreover, in [notebook 02LatentSemanticIndexing](02LatentSemanticIndexing.ipynb) LSI based topic extraction and document clustering have also been introduced by a small playground example.

The current notebook applies these concepts to a real corpus of RSS-Feeds, which has been generated and accessed in previous notebooks of this lecture:

## Read documents from a corpus 

The contents of the RSS-Fedd corpus are imported by NLTK's `CategorizedPlaintextCorpusReader` as already done in previous notebooks of this lecture:

#!pip install wordcloud

from nltk.corpus.reader import CategorizedPlaintextCorpusReader
from nltk.corpus import stopwords
stopwordlist=stopwords.words('german')
from wordcloud import WordCloud

rootDir="../01access/GERMAN"
filepattern=r"(?!\.)[\w_]+(/RSS/FeedText/)[\w-]+/[\w-]+\.txt"
#filepattern=r"(?!\.)[\w_]+(/RSS/FullText/)[\w-]+/[\w-]+\.txt"
catpattern=r"([\w_]+)/.*"
rssreader=CategorizedPlaintextCorpusReader(rootDir,filepattern,cat_pattern=catpattern)

singleDoc=rssreader.paras(categories="TECH")[0]
print("The first paragraph:\n",singleDoc)
print("Number of paragraphs in the corpus: ",len(rssreader.paras(categories="TECH")))

techdocs=[[w.lower() for sent in singleDoc for w in sent if (len(w)>1 and w.lower() not in stopwordlist)] for singleDoc in rssreader.paras(categories="TECH")]
print("Number of documents in category Tech: ",len(techdocs))

generaldocs=[[w.lower() for sent in singleDoc for w in sent if (len(w)>1 and w.lower() not in stopwordlist)] for singleDoc in rssreader.paras(categories="GENERAL")]
print("Number of documents in category General: ",len(generaldocs))

alldocs=techdocs+generaldocs
print("Total number of documents: ",len(alldocs))

### Remove duplicate news

def removeDuplicates(nestedlist):
    listOfTuples=[tuple(liste) for liste in nestedlist]
    uniqueListOfTuples=list(set(listOfTuples))
    return [list(menge) for menge in uniqueListOfTuples]

techdocs=removeDuplicates(techdocs)
generaldocs=removeDuplicates(generaldocs)
alldocs=removeDuplicates(alldocs)
print("Number of unique documents in category Tech: ",len(techdocs))
print("Number of unique documents in category General: ",len(generaldocs))
print("Total number of unique documents: ",len(alldocs))

alltechString=" ".join([w for doc in techdocs for w in doc])
print(len(alltechString))
allgeneralString=" ".join([w for doc in generaldocs for w in doc])
print(len(allgeneralString))

wordcloudTech=WordCloud().generate(alltechString)
wordcloudGeneral=WordCloud().generate(allgeneralString)
%matplotlib inline
import matplotlib.pyplot as plt
plt.figure(figsize=(20,18))
plt.title("Tech News")
plt.subplot(1,2,1)
plt.imshow(wordcloudTech, interpolation='bilinear')
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(wordcloudGeneral, interpolation='bilinear')
plt.title("General News")
plt.axis("off")

## Gensim-representation of imported RSS-feeds 

from gensim import corpora, models, similarities
dictionary = corpora.Dictionary(alldocs)
dictionary.save('feedwordsDE.dict') # store the dictionary, for future reference
#print(dictionary.token2id)
print(len(dictionary))

firstdoc=techdocs[0]
print(firstdoc)
firstVec = dictionary.doc2bow(firstdoc)
print("Sparse BoW representation of single document: %s"%firstVec)
w1='champions'
w2='league'
w3='heute'
print("Index of word %s is %d"%(w1,dictionary.token2id[w1]))
print("Index of word %s is %d"%(w2,dictionary.token2id[w2]))
print("Index of word %s is %d"%(w3,dictionary.token2id[w3]))

Sparse BoW representation of entire tech-corpus and entire general-news-corpus: 

techcorpus = [dictionary.doc2bow(doc) for doc in techdocs]
generalcorpus = [dictionary.doc2bow(doc) for doc in generaldocs]

print(generaldocs[:3])

## Find similiar documents

index = similarities.SparseMatrixSimilarity(techcorpus, num_features=len(dictionary))

sims = index[firstVec]
#print(list(enumerate(sims)))
simlist = sims.argsort()
print(simlist)
mostSimIdx=simlist[-2]

print("Refernce document is:\n",firstdoc)
print("Most similar document:\n",techdocs[mostSimIdx])

## Find topics by Latent Semantic Indexing (LSI)
### Generate tf-idf model of corpus

tfidf = models.TfidfModel(techcorpus)
corpus_tfidf = tfidf[techcorpus]
print("Display TF-IDF- Model of first 2 documents of the corpus")
for doc in corpus_tfidf[:2]:
    print(doc)

### Generate LSI model from tf-idf model

techdictionary = corpora.Dictionary(techdocs)

NumTopics=50
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=NumTopics) # initialize an LSI transformation

corpus_lsi = lsi[corpus_tfidf]

Display first 10 topics:

lsi.print_topics(10)

### Determine the most relevant documents for a selected topic

Generate a numpy array `docTopic`. The entry in row $i$, column $j$ of this array is the relevance value for topic $j$ in document $i$.

import numpy as np
numdocs= len(corpus_lsi)
docTopic=np.zeros((numdocs,NumTopics))

for d,doc in enumerate(corpus_lsi): # both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly
    for t,top in enumerate(doc):
        docTopic[d,t]=top[1]
print(docTopic.shape)
print(docTopic)

Select an arbitrary topic-id and determine the documents, which have the highest relevance value for this topic:

topicId=7 #select an arbitrary topic-id
topicRelevance=docTopic[:,topicId]

docsoftopic= np.array(topicRelevance).argsort()
relevanceValue= np.sort(topicRelevance)
print(docsoftopic) #most relevant document for selected topic is at first position
print(relevanceValue) #highest relevance document/topic-relevance-value is at first position

TOP=3
print("Selected Topic:\n",lsi.show_topic(topicId))
print("#"*50)
print("Docs with the highest negative value w.r.t the selected topic")
for idx in docsoftopic[:TOP]:
    print("-"*20)
    print(idx,"\n",techdocs[idx])
print("#"*50)
print("Docs with the highest positive value w.r.t the selected topic")
for idx in docsoftopic[-TOP:]:
    print("-"*20)
    print(idx,"\n",techdocs[idx])

import gensim


lda = gensim.models.ldamodel.LdaModel(corpus_tfidf, num_topics=20, id2word = dictionary)

#!pip install pyLDAvis

import pyLDAvis.gensim as gensimvis
import pyLDAvis

vis_en = gensimvis.prepare(lda, corpus_tfidf, dictionary)
pyLDAvis.display(vis_en)

