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

vector = word_vectors['computer']  # numpy vector of a word
print(vector.shape)
print(vector[:10])

np.sqrt(np.sum(np.square(vector)))

vector = word_vectors.word_vec('office', use_norm=True)
print(vector.shape)
print(vector[:10])

np.sqrt(np.sum(np.square(vector)))

