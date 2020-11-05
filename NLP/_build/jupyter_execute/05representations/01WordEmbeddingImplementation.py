# Applying Word-Embeddings


There are different options to work with Word-Embeddings:
1. Trained Word-Embeddings can be downloaded from the web. These Word-Embeddings differ in
    * the method, e.g. Skipgram, CBOW, GloVe, fastText 
    * in the hyperparameter applied for the selected method, e.g. context-length
    * in the corpus, which has been applied for training
2. By applying packages such as [gensim](https://radimrehurek.com/gensim/) word-embeddings can easily be trained from an arbitrary collection of texts 
3. Training of a word embedding can be integrated into an end-to-end neural network for a specific application. For example, if a Deep-Nerual-Network shall be learned for document-classification, the first layer in this network can be defined, such that it learns a task-specific word-embedding from the given document-classification-training-data.

In this notebook option 1 and 2 are demonstrated. Option 3 is applied in a later lecture

## Apply Pre-Trained Word-Embeddings
### FastText


The [FastText project](https://fasttext.cc) provides word-embeddings for 157 different languages, trained on [Common Crawl](https://commoncrawl.org/) and [Wikipedia](https://www.wikipedia.org/). These word embeddings can easily be downloaded and imported to Python. The `KeyedVectors`-class of [gensim](https://radimrehurek.com/gensim/) can be applied for the import. This class also provides many useful tools, e.g. an index to fastly find the vector of an arbitrary word or function to calculate similarities between word-vectors. Some of these tools will be demonstrated below: 

After downloading word embeddings from [FastText](https://fasttext.cc/docs/en/english-vectors.html) they can be imported into a `KeyedVectors`-object from gensim as follows:

from gensim.models import KeyedVectors
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Creating the model
#en_model = KeyedVectors.load_word2vec_format('/Users/maucher/DataSets/Gensim/FastText/Gensim/FastText/wiki-news-300d-1M.vec')
#en_model = KeyedVectors.load_word2vec_format(r'C:\Users\maucher\DataSets\Gensim\Data\Fasttext\wiki-news-300d-1M.vec\wiki-news-300d-1M.vec') #path on surface
#en_model = KeyedVectors.load_word2vec_format('/Users/maucher/DataSets/Gensim/FastText/fasttextEnglish300.vec')
en_model = KeyedVectors.load_word2vec_format('/Users/johannes/DataSets/Gensim/FastText/fasttextEnglish300.vec') # path on iMAC

The number of vectors and their length can be accessed as follows:

# Printing out number of tokens available
print("Number of Tokens: {}".format(en_model.vectors.shape[0]))

# Printing out the dimension of a word vector 
print("Dimension of a word vector: {}".format(en_model.vectors.shape[1]))

The first 20 words in the index:

en_model.wv.index2word[:20]

The first 10 components of the word-vector for *evening*:

en_model["evening"][:10]

The first 10 components of the word-vector for *morning*:

en_model["morning"][:10]

The similarity between *evening* and *morning*:

similarity = en_model.similarity('morning', 'evening')
similarity

The 20 words, which are most similar to word *wood*:

en_model.most_similar("wood",topn=20)

### GloVe
As described [before](05representations.md) GloVe constitutes another method for calculating Word-Embbedings. Pre-trained GloVe vectors can be downloaded from
[Glove](https://nlp.stanford.edu/projects/glove/) and imported into Python. However, gensim already provides a downloader for several word-embeddings, including GloVe embeddings of different length and different training-data. 

The corpora and embeddings, which are available via the gensim downloader, can be queried as follows:

import gensim.downloader as api

api.info(name_only=True)

We select the GloVe word-embeddings `glove-wiki-gigaword-100` for download: 

word_vectors = api.load("glove-wiki-gigaword-100")  # load pre-trained word-vectors from gensim-data

type(word_vectors)

As can be seen in the previous output, the downloaded data is available as a `KeyedVectors`-object. Hence the same methods can now be applied as in the case of the FastText - Word Embedding in the previous section. In the sequel we will apply not only the methods used above, but also new ones.

Word analogy questions like *man is to king as woman is to ?* can be solved as in the code cell below:

result = word_vectors.most_similar(positive=['woman', 'king'], negative=['man'])
print("{}: {:.4f}".format(*result[0]))

Outliers within sets of words can be determined as follows:

print(word_vectors.doesnt_match("breakfast cereal dinner lunch".split()))

Similiarity between a pair of words:

similarity = word_vectors.similarity('woman', 'man')
print(similarity)

Most similar words to *cat*:

word_vectors.most_similar("cat",topn=20)

Similarity between sets of words:

sim = word_vectors.n_similarity(['sushi', 'shop'], ['japanese', 'restaurant'])
print("{:.4f}".format(sim))

First 10 components of word vector for *computer*:

vector = word_vectors['computer']  # numpy vector of a word
print(vector.shape)
print(vector[:10])

The magnitude of the previous word-vector.

np.sqrt(np.sum(np.square(vector)))

As can be seen in the previous code cell the vectors are not normalized to unique length. However, if the argument `use_norm` is enabled, the resulting vectors are normalized:

vector = word_vectors.word_vec('office', use_norm=True)
print(vector.shape)
print(vector[:10])

np.sqrt(np.sum(np.square(vector)))

## Visualisation of Word-Vectors

Typical lengths of DSM word vectors are in the range between 50 and 300. In the FastText example above vectors of length 300 have been applied. The applied GloVe vectors had a length of 100. In any case they can not directly be visualised. However, methods to reduce the dimensionality of vectors in such a way, that their overall spatial distribution is maintained as much as possible can be applied to transform word vectors into 2-dimensional space. In the code cells below this is demonstrated by applying **TSNE**, the most prominent technique to transform word-vectors into 2-dimensional space: 

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

tsneModel=TSNE(n_components=2,random_state=0)
np.set_printoptions(suppress=True)
model2d=tsneModel.fit_transform(word_vectors[word_vectors.index2word[300:600]])

#%matplotlib inline
plt.figure(figsize=(19,14))
idx=0
for a in model2d[:300]:
    w=word_vectors.wv.index2word[300+idx]
    plt.plot(a[0],a[1],'r.')
    plt.text(a[0],a[1],w)
    idx+=1
plt.show()

This simple visualisation already indicates, that the vectors of similar words are closer to each other, than the vectors of unrelated words.

## Train Word Embedding
In this section it is demonstrated how [gensim](https://radimrehurek.com/gensim/) can be applied to train a Word2Vec (either CBOW or Skipgram) embedding from an arbitrary corpus. In this demo the applied training corpus is the complete English Wikipedia dump.   

### Download and Extract Wikipedia Dump

Wikipedia dumps can be downloaded from [here](https://dumps.wikimedia.org/other/wikibase/wikidatawiki/). After downloading the dump the most convenient way to extract and clean the text is to apply the [WikiExtractor](https://github.com/attardi/wikiextractor). This tool generates plain text from a Wikipedia database dump, discarding any other information or annotation present in Wikipedia pages, such as images, tables, references and lists.
The output is stored in a number of files of similar size in a given directory.

The class `MySentences`, as defined in the following code-cell, extracts from all directories and files under `dirnameP` the sentences in a format, which can be processed by the applied gensim model.

import os,logging
from gensim.models import word2vec

class MySentences(object):
    def __init__(self, dirnameP):
        self.dirnameP = dirnameP
 
    def __iter__(self):
        for subdir in os.listdir(self.dirnameP):
            print(subdir)
            if subdir==".DS_Store":
                continue
            subdirpath=os.path.join(self.dirnameP,subdir)
            print(subdirpath)
            for fname in os.listdir(subdirpath):
                if fname[:4]=="wiki":
                    for line in open(os.path.join(subdirpath, fname)):
                        linelist=line.split()
                        if len(linelist)>3 and linelist[0][0]!="<":
                            yield [w.lower().strip(",."" \" () :; ! ?") for w in linelist]

The path to the directory, which contains the entire extracted Wikipedia dump is configured and the subdirectories under this path are listed:

#parentDir="C:\\Users\\maucher\\DataSets\\Gensim\\Data\\wiki_dump_extracted"
parentDir="/Users/johannes/DataSets/wikiextractor/text" #path on iMAC
#parentDir="C:\Users\Johannes\DataSets\Gensim\Data\wiki_dump_extracted"
dirlistParent= os.listdir(parentDir)
print(dirlistParent)

### Training or Loading of a CBOW model
In the following code cell a name for the word2vec-model is specified. If the specified directory already contains a model with the specified name, it is loaded. Otherwise, it is generated and saved under the specified name. A **skipgram-model** can be generated in the same way. In this case `model = word2vec.Word2Vec(sentences,size=200,sorted_vocab=1)` has to be replaced by `model = word2vec.Word2Vec(sentences,size=200,sorted_vocab=1,sg=1)`. 
See [gensim model.Word2Vec documentation](https://radimrehurek.com/gensim/models/word2vec.html) for the configuration of more parameters. 

> Note that the training of this model takes several hours. If you like to generate a much smaller model from a smaller corpus (English!) you can download the text8 corpus from [http://mattmahoney.net/dc/text8.zip](http://mattmahoney.net/dc/text8.zip), extract it and replace the code in the following code-cell by this:
```
sentences = word2vec.Text8Corpus('C:\\Users\\maucher\\DataSets\\Gensim\\Data\\text8')
model = word2vec.Word2Vec(sentences,size=200)
```

modelName="/Users/johannes/DataSets/wikiextractor/models/wikiEng20201007.model"
try:
    model=word2vec.Word2Vec.load(modelName)
    print("Already existing model is loaded")
except:
    print("Model doesn't exist. Training of word2vec model started.")
    sentences = MySentences(parentDir) # a memory-friendly iterator
    model = word2vec.Word2Vec(sentences,size=200,sorted_vocab=1)
model.init_sims(replace=True)
model.save(modelName)

type(model)

In the code cell above the Word2Vec model has either been created or loaded. For the returned object of type `Word2Vec` basically the same functions are available as for the pretrained FastText and GloVe word embeddings in the sections above. 

For example the most similar words for *cat* are:

model.most_similar("cat",topn=20)

For the trained Word2Vec-model also parameters, which describe training, corpus and the model itself, can be accessed, as demonstrated below: 

print("Number of words in the corpus used for training the model: ",model.corpus_count)
print("Number of words in the model: ",len(model.wv.index2word))
print("Time [s], required for training the model: ",model.total_train_time)
print("Count of trainings performed to generate this model: ",model.train_count)
print("Length of the word2vec vectors: ",model.vector_size)
print("Applied context length for generating the model: ",model.window)

