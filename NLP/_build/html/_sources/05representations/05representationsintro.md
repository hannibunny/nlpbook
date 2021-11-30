# Vector Representations of Words and Documents

As described in the {doc}`../intro`, many NLP tasks can either be solved in a rule-based or in a data-based approach. Data-based approaches increasingly yield better results than rule-based approaches. At the heart of data-based approaches is a Machine Learning algorithm, which learns a model from available training data. Such models can be learned for a wide variety of e.g. for Document Classification, Sentiment Analysis, Named-Entity-Recognition, Intent-Recognition, Question-Answering, Automatic translation and many others.

All ML-algorithms require a numeric representation at their input, usually a fixed-length numeric vector. In the context of NLP the input is usually text. The crucial question is then:

```{admonition} Question:
How to represent texts, i.e. sequences of words, punctuation marks..., as a numeric vector of constant length?
```

In this section first the general notion of **Vector Space Model** is introduced. Then the most common type of vector space model for text, the so called **Bag-Of-Word (BoW)** model is described. In the extreme case of single-word-texts the BoW melts down to the **One-Hot-Encoding** of words. The pros and cons of these conventional representations are discussed. Another method to represent words as numerical vectors is constituted by **Distributional Semantic Models (DSMs)**. The currently very popular **Word Embeddings** belong to the class of DSMs. **Word Embeddings** are numerical word representations, which are learned by *Neural Networks*. Even though they have been considered before, the 2013 milestone paper {cite}`NIPS2013_5021` introduced two very efficient methods to learn meaningful Word Embeddings. Since then Word Embeddings have revolutionized NLP.


