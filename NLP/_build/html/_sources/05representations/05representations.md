# Vector Representations of Words and Documents

As described in the {doc}`intro`, many NLP tasks can either be solved in a rule-based or in a data-based approach. Data-based approaches increasingly yield better results than rule-based approaches. At the heart of data-based approaches is a Machine Learning algorithm, which learns a model from available training data. This model can then be applied e.g. for Document Classification, Sentiment Analysis, Named-Entity-Recognition, Intent-Recognition, Automatic translation and many other NLP tasks.

All ML-algorithms require a numeric representation at their input, usually a fixed-length numeric vector. In the context of NLP the input is usually text. The crucial question is then:

```{admonition} Question:
How to represent texts, i.e. sequences of words, punctuation marks..., as a numeric vector of constant length?
```

In this section first the general notion of **Vector Space Model** is introduced. Then the most common type of vector space model for text, the so called **Bag-Of-Word (PoW)** model is described. In the extreme case of single-word-texts the BoW melts down to the **One-Hot-Encoding** of words. The pros and cons of these conventional representations are discussed. Another method to represent words as numerical vectors is constituted by **Distributional Semantic Models (DSMs)**. The currently very popular **Word Embeddings** belong to the class of DSMs. **Word Embeddings** are numerical word representations, which are learned by *Neural Networks*. Even though they have been considered before, the 2013 milestone paper {cite}`NIPS2013_5021` introduced two very efficient methods to learn meaningful Word Embeddings. Since then Word Embeddings have revolutionized NLP.

## Vector Space Model

Vector Space Models map arbitrary inputs to numeric vectors of fixed length. For a given task, you are free to define a set of $N$ relevant features, which can be extracted from the input. Each of the $N$-feature extraction functions returns how often the corresponding feature appears in the input. Each component of the vector-representation belongs to one feature and the value at this component is the count of this feature in the input.

```{admonition} Example
:class: dropdown
Assume that your task is to classify texts into the classes *poetry* and *scientific paper*. The classifier shall be learned by a Machine Learning Algorithm which requires fixed-length numeric vectors at it's input. You think about relevant features and come to the conclusion, that

1. the average length of sentences (in words)
2. the number of proper names
3. the number of adjectives

may be relevant features. Then all input-texts, independent of their length can be mapped to vectors of length $N=3$, whose components are the frequencies of this features in the text. E.g. the text

`Mary loves the kind and handsome little boy. His name is Peter and he lived next door to Mary's jealous friend Anne.`


maps into the vector 

$$
(11,4,4) 
$$
```

### Bag-of-Word Model

### Bag-of-Word Variants

### One-Hot-Encoding of Words


## Distributional Semantic Models

### Count-based DSM

### Prediction-based DSM 

