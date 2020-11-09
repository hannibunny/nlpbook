## Topic Extraction

As described in section {ref}`dsm-label`, word-embeddings provide an efficient way to model semantic simliarity between words. A DSM-based word-embedding maps words, which are semantically, related to similar vectors. Now, we are interested in semantically related documents. **We like to find out which documents are semantically related in the sense that they refer to similar topics.** We already know, that documents can be modelled by BoW-vectors and that we can determine similarity between documents by just calculating the similarity (or distance) between the corresponding BoW-vectors. However, the problem with this approach is, that we can describe the same topic with different words. For example the sentences *The president visited Europe* means essentially the same as the sentence *The head of state had a trip to EU capitals*. Applying BoW both sentences would have totally different vectors, even though they describe the same topic. 

In NLP, the goals of **topic extraction** are 

* given a corpus of documents, find out which topics are discussed in these documents
* find out which documents describe which topics
* *topics* may be better features than words. Therefore topic extraction may be applied as preprocessing for other tasks, such as text-classification.

Algorithms for topic extraction are e.g. *Latent Semantic Indexing (LSI), Non-Negative Matrix Factorisation (NNMF)* or *Latent Dirichlet Allocation*. In this notebook LSI is described.

