# Document models and similarity

- Author:      Johannes Maucher
- Last update: 05.11.2020

This notebook demonstrates how documents can be described in a vector space model. Applying this type of model

1. similarities between documents 
2. similarities between documents and a query 

can easily be calculated.

## Read documents from a textfile 

It is assumed that a set of documents is stored in a textfile, as e.g. in [MultiNewsFeeds2014-09-12.txt](../Data/MultiNewsFeeds2014-09-12.txt). The individual documents are separated by line-break. In this case the documents can be assigned to the list _listOfNews_ as follows:

filename="../Data/MultiNewsFeeds2014-09-12.txt"
#filename="../Data/MultiNewsFeeds2016-10-14.txt"
listOfNews=[]
with open(filename,"r",encoding="utf-8") as fin:
    for line in fin:
        line = line.strip()
        print(line)
        listOfNews.append(line)
print("Number of Lines:  ",len(listOfNews))
fin.close()

## Split documents into words, normalize and remove stopwords
In _listOfNews_ each document is stored as a single string variable. Each of these document-strings is now split into a set of words. All words are transformed to a lower-case representation and stop-words are removed.

from nltk.corpus import stopwords
stopwordlist=stopwords.words('german')
docWords = [[word.strip('?!.:",') for word in document.lower().split() 
             if word.strip('?!.:",') not in stopwordlist] for document in listOfNews]
#print(docWords)

Display the list of words of the first 5 documents:

idx=0
for doc in docWords[:5]:
    print('------ document %d ----------'%idx)
    for d in doc:
        print(d)
    idx+=1

## Generate Dictionary
The elements of the list _docWords_ are itself lists. Each of these lists contains all relevant words of a document. The set of all relevant words in the document collection, i.e. relevant words, which appear in at least one document, are stored in a [gensim-dictionary](https://radimrehurek.com/gensim/corpora/dictionary.html). In the dictionary to each of the relevant words an unique integer ID is assigned: 

from gensim import corpora, models, similarities
dictionary = corpora.Dictionary(docWords)
dictionary.save('multiNews.dict') # store the dictionary, for future reference
print(dictionary.token2id)

print("Total number of documents in the dictionary: ",dictionary.num_docs)
print("Total number of corpus positions: ",dictionary.num_pos)
print("Total number of non-zeros in the BoW-Matrix: ",dictionary.num_nnz)
print("Total number of different words in the dictionary: ",len(dictionary))

## Bag of Word (BoW) representation
Now arbitrary text-strings can be efficiently represented with respect to this dictionary. E.g. the code snippet below demonstrates how the text string _"putin beschützt russen"_ is represented as a list of tuples. The first element of such a tuple is the dictionary index of a word in the text-string and the second number defines how often this word occurs in the text-string. The list contains only tuples for words which occur in the text-string and in the dictionary. This representation is called **sparse Bag of Word** representation (sparse because it contains only the non-zero elements).

newDoc = "putin beschützt russen"
newVec = dictionary.doc2bow(newDoc.lower().split())
print("Sparse BoW representation of %s: %s"%(newDoc,newVec))
for idx,freq in newVec:
    print("Index %d refers to word %s. Frequency of this word in the document is %d"%(idx,dictionary[idx],freq))

From this output we infer, that 

* the word at index 189 in the dictionary, which is _beschützt_ , appears once in the text-string _newDoc_
* the word at index 397 in the dictionary, which is _putin_ , appears once in the text-string _newDoc_
* the word at index 729 in the dictionary, which is _russen_ , appears once in the text-string _newDoc_ .

The text-string _"schottland stimmt ab"_ is represented as a list of 2 tuples (see code snippet below). The first says, that the word at index 229 ( _ab_ ) appears once, the second tuple says that the word at index _807_ ( _schottland_ ) also appears once in the text-string. Since the word _stimmt_ does not appear in the dictionary, there is no corresponding tuple for this word in the list.  


newDoc2 = "schottland stimmt ab"
newVec2 = dictionary.doc2bow(newDoc2.lower().split())
print("Sparse BoW representation of %s: %s"%(newDoc2,newVec2))
for idx,freq in newVec2:
    print("Index %d refers to word %s. Frequency of this word in the document is %d"%(idx,dictionary[idx],freq))

## Efficient Corpus Representation
A corpus is a collection of documents. Such corpora may be annotated with meta-information, e.g. each word is tagged with its part of speech (POS-Tag). In this notebook, the list _docWords_, is a corpus without any annotations. So far this corpus has been applied to build the dictionary. In practical NLP tasks corpora are usually very large and therefore require an efficient representation. Using the already generated dictionary, each document (list of relevant words in a document) in the list _docWords_ can be transformed to its sparse BoW representation. 

corpus = [dictionary.doc2bow(doc) for doc in docWords]
corpora.MmCorpus.serialize('multiNews.mm', corpus)
print("------------------------- First 10 documents of the corpus ---------------------------------")
idx=0
for d in corpus[0:10]:
    print("-------------document %d ---------------" %idx)
    print(d)
    idx+=1

## Similarity Analysis
Typical information retrieval tasks include the task of determining similarities between collections of documents or between a query and a collection of documents. Using gensim a fast similarity calculation and search is supported. For this, first a **cosine-similarity-index** of the given corpus is calculated as follows: 

index = similarities.SparseMatrixSimilarity(corpus, num_features=len(dictionary))

Now, assume that for a given query, e.g. _"putin beschützt russen"_ the best matching document in the corpus must be determined. The sparse BoW representation of this query has already been calculated and stored in the variable _newVec_. The similarity between this query and all documents in the corpus can be calculated as follows:

sims = index[newVec]
print(list(enumerate(sims)))

The tuples in this output contain as first element the index of the document in the corpus. The second element is the cosine-similarity between this corpus-document and the query. 

In order to get a sorted list of increasingly similar documents, the `argsort()`-method can be applied as shown below. The last value in this list is the index of the most similar document:

print(sims.argsort())

In this example _document 43_ best matches to the query. The cosine-similarity between the query and _document 43_ is _0.2357_.

Question: Manually verify the calculated similiarity value between the query and _document 43_.

In the same way the similarity between documents in the corpus can be calculated. E.g. the similiarity between _document 1_ and all other documents in the corpus is determined as follows:

sims = index[corpus[1]]
print((list(enumerate(sims))))
print(sims.argsort())

Thus _document 15_ is the most similar document to _document 1_. As can easily be verified both documents refer to the same topic (crisis in ukraine).

## TF-IDF representation
So far in the BoW representation of the documents the _term frequency (tf)_ has been applied. This value measures how often the term (word) appears in the document. If document similarity is calculated on such tf-based BoW representation, common words which appear quite often (in many documents) but have low semantic focus have a strong impact on the similarity-value. In most cases this is a drawback, since similarity should be based on terms with a high semantic focus. Such semantically meaningful words usually appear only in a few documents. The _term frequency inversed document frequency measure (tf-idf)_ does not only count the frequency of a term in a document, but weighs those terms stronger, which occur only in a few documents of the corpus. 

In _gensim_ the _tfidf_ - model of a corpus can be calculated as follows:

tfidf = models.TfidfModel(corpus)

The _tf-idf_-representation of the first 3 documents in the corpus are:

idx=0
for d in corpus[:3]:
    print("-------------tf-idf BoW of document %d ---------------" %idx)
    print(tfidf[d])
    idx+=1

In this representation the second element in the tuples is not the term frequency, but the _tfidf_. Note that default configuration of [tf-idf in gensim](http://radimrehurek.com/gensim/models/tfidfmodel.html) calculates tf-idf values such that each document-vector has a norm of _1._ The tfidf-model without normalization is generated at the end of this notebook.

Question: Find the maximum tf-idf value in these 3 documents. To which word does this maximum value belong? How often does this word occur in the document?

The _tf-idf_-representation of the text-string _"putin beschützt russen"_ is determined as follows:

newVecTfIdf = tfidf[newVec]
print("tf BoW representation of %s is:\n %s"%(newDoc,newVec))
print("tf-idf BoW representation of %s is:\n %s"%(newDoc,newVecTfIdf))

Question: Explain the different values in the tfidf BoW representation _newVecTfIdf_. 

**TF-IDF-Model without normalization:**

tfidfnoNorm = models.TfidfModel(corpus,normalize=False)

Display tf-idf BoW of first 3 documents:

idx=0
for d in corpus[:3]:
    print("-------------tf-idf BoW of document %d ---------------" %idx)
    print(tfidfnoNorm[d])
    idx+=1

Verify the tf-idf-values as calculated in the code-cell above, by own tf-idf-formula:

import numpy as np
tf=1 #term frequency
NumDocs=dictionary.num_docs #number of documents
df=1 #number of documents in which the word appears
tfidf=tf*np.log2(float(NumDocs)/df)
print(tfidf)

## Tokenisation and Document models with Keras
This section demonstrates how [Keras](https://keras.io/api/preprocessing/text/) can be applied for tokenisation and BoW document-modelling. I.e. no new techniques are introduced here. Instead it is shown how Keras can be applied to implement already known procedures. This is usefunl, because Keras will be applied later on to implement Neural Networks.

### Tokenizsation

#### Text collections as lists of strings
Tokens are atomic text elements. Depending on the NLP task and the selected approach to solve this task, tokens can either be
* characters
* words (uni-grams)
* n-grams

Single texts are often represented as variables of type `string`. Collections of texts are then represented as lists of strings.

Below, a collection of 3 texts is generated as a list of `string`-variables:

text1="""Florida shooting: Nikolas Cruz confesses to police Nikolas Cruz is said
to have killed 17 people before escaping and visiting a McDonalds."""
text2="""Winter Olympics: Great Britain's Dom Parsons wins skeleton bronze medal
Dom Parsons claims Great Britain's first medal of 2018 Winter Olympics with bronze in the men's skeleton."""
text3="""Theresa May to hold talks with Angela Merkel in Berlin
The prime minister's visit comes amid calls for the UK to say what it wants from Brexit."""

print(text1)

textlist=[text1,text2,text3]

#### Keras class Tokenizer

In Keras methods for preprocessing texts are contained in `keras.preprocessing.text`. From this module, we apply the `Tokenizer`-class to 
* transform words to integers, i.e. generating a word-index
* represent texts as sequences of integers
* represent collections of texts in a Bag-of-Words (BOW)-matrix

from keras.preprocessing import text

Generate a `Tokenizer`-object and fit it on the given list of texts:

tokenizer=text.Tokenizer()
tokenizer.fit_on_texts(textlist)

The `Tokenizer`-class accepts a list of arguments, which can be configured at initialisation of a `Tokenizer`-object. The default-values are printed below:

print("Configured maximum number of words in the vocabulary: ",tokenizer.num_words) #Maximum number of words to regard in the vocabulary
print("Configured filters: ",tokenizer.filters) #characters to ignore in tokenization
print("Map all characters to lower case: ",tokenizer.lower) #Mapping of characters to lower-case
print("Tokenizsation on character level: ",tokenizer.char_level) #whether tokens are words or characters

print("Number of documents: ",tokenizer.document_count)

Similar as the `dictionary` in gensim (see above), the Keras `Tokenizer` provides a word-index, which uniquely maps each word to an integer:

print("Index of words: ",tokenizer.word_index)

The method `word_docs()` returns for each word the number of documents, in which the word appears:

print("Number of docs, in which word appears: ",tokenizer.word_docs)

### Represent texts as sequences of word-indices:

The following representation of texts as sequences of word-indicees is a common input to Neural Networks implemented in Keras.

textSeqs=tokenizer.texts_to_sequences(textlist)
for i,ts in enumerate(textSeqs):
    print("text %d sequence: "%i,ts)

### Represent text-collection as binary BoW:
A Bag-Of-Words representation of documents contains $N$ rows and $|V|$ columns, where $N$ is the number of documents in the collection and $|V|$ is the size of the vocabulary, i.e. the number of different words in the entire document collection.

The entry $x_{i,j}$ of the BoW-Matrix indicates the **relevance of word $j$ in document $i$**.

In this lecture 3 different types of **word-relevance** are considered:

1. **Binary BoW:** Entry $x_{i,j}$ is *1* if word $j$ appears in document $i$, otherwise 0.
2. **Count-based BoW:** Entry $x_{i,j}$ is the frequency of word $j$ in document $i$.
3. **Tf-idf-based BoW:** Entry $x_{i,j}$ is the tf-idf of word $j$ with respect to document $i$.

The BoW-representation of texts is a common input to conventional Machine Learning algorithms (not Neural Netorks like CNN and RNN).

#### Binary BoW

print(tokenizer.texts_to_matrix(textlist))

#### Count-based BoW
Represent text-collection as BoW with word-counts:

print(tokenizer.texts_to_matrix(textlist,mode="count"))

#### Tf-idf-based BoW
In the BoW representation above the term frequency (tf) has been applied. This value measures how often the term (word) appears in the document. If document similarity is calculated on such tf-based BoW representation, common words which appear quite often (in many documents) but have low semantic focus, have a strong impact on the similarity-value. In most cases this is a drawback, since similarity should be based on terms with a high semantic focus. Such semantically meaningful words usually appear only in a few documents. The term frequency inversed document frequency measure (tf-idf) does not only count the frequency of a term in a document, but weighs those terms stronger, which occur only in a few documents of the corpus.

print(tokenizer.texts_to_matrix(textlist,mode="tfidf"))

