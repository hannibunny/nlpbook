# Vector Representations of Words and Documents

As described in the {doc}`../intro`, many NLP tasks can either be solved in a rule-based or in a data-based approach. Data-based approaches increasingly yield better results than rule-based approaches. At the heart of data-based approaches is a Machine Learning algorithm, which learns a model from available training data. This model can then be applied e.g. for Document Classification, Sentiment Analysis, Named-Entity-Recognition, Intent-Recognition, Automatic translation and many other NLP tasks.

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

 In the general case the vector space model implies a vector, whose components are the frequencies of pre-defined features in the given input. In the special case of text (=documents), a vector space model is applied, where the features are defined to be all words of the vocabulary $V$. I.e. each component in the resulting vector corresponds to a word $w \in V$ and the value of the component is the frequency of this word in the given document. This vector space model for texts is the so called **Bag of Words (BoW)** model and the frequency of a word in a given document is denoted **term-frequency**. Accordingly a set of documents is modelled by a **Bag of Words matrix**, whose rows belong to documents and whose columns belong to words. 
 
     
```{admonition} Example: Bag of Word matrix
:class: dropdown

Assume, that the given playground-corpus contains only two documents

* Document 1: *not all kids stay at home*
* Document 2: *all boys and girls stay not at home*

The BoW model of these documents is then

|            | all | and | at   | boys | girls | home | kids | not  | stay |
|------------|-----|-----|------|------|-------|------|------|------|------|
| Document 1 | 1   | 0   | 1    | 0    | 0     | 1    | 1    | 1    | 1    |
| Document 2 | 1   | 1   | 1    | 1    | 1     | 1    | 0    | 1    | 1    |
 
In this example the words in the matrix have been alphabetically ordered. This is not necessary.
```

### Bag-of-Word Variants

The entries of the BoW-matrix, as introduced above, are the **term-frequencies**. I.e. the entry in row $i$, column $j$, $tf(i,j)$ determines how often the term (word) of column $j$, appears in document $j$. 

Another option is the **binary BoW**. Here, the binary entry in row $i$, column $j$ just indicates if term $j$ appears in document $i$. The entry has value 1 if the term appears at least once, otherwise it is 0.

**TF-IDF BoW:** The drawback of using term-frequency $tf(i,j)$ as matrix entries is that all terms are weighted similarly, in particular rare words such as words with a **strong semantic focus** are weighted in the same way as very frequent words, such as articles. *TF-IDF* is a weighted term frequency (TF). The weights are the *inverse document frequencies (IDF)*. Actually, there are different definitions for the calculation of TF-IDF. A common definition is

$$
\mbox{tf-idf}(i,j) = tf(i,j) \cdot \log(\frac{N}{df_j}),
$$

where $tf(i,j)$ is the frequency of term $j$ in document $i$, $N$ is the total number of documents and $df_j$ is the number of documents, which contain term $j$. For words, which occure in all documents 

$$
\log(\frac{N}{df_j}) = \log(\frac{N}{N}) = 0,
$$

i.e. such words are disregarded in a BoW with TF-IDF entries. Otherwise, words with a very strong semantic focus usually appear in only a few documents. Then the small value of $df_j$ yields a low *IDF*, i.e. the term-frequency of such a word is weighted strongly.

  

### One-Hot-Encoding of Words

In the extreme case of *documents*, which contain only a single word, the corresponding *tf*-based BoW-vector, has only one component of value 1 (in the column, which belongs to this word), all other entries are zero. This is actually a common conventional numeric encoding of words, the so called *One-Hot-Encoding*.


```{admonition} Example: One-Hot-Encoding of words
:class: dropdown

Assume, that the entire Vocabular is 

$$
V=(\mbox{all, and, at, boys, girls, home, kids, not, stay}).
$$

A possible One-Hot-Encoding of these words is then

|       |   |   |   |   |   |   |   |   |   |
|-------|---|---|---|---|---|---|---|---|---|
| all   | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| and   | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| at    | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| boys  | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| girls | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| home  | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| kids  | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| not   | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| stay  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |

```

### BoW-based document similarity

|            	| term 1 	| term 2 	| term 3 	| term 4 	|
|------------	|:------:	|:------:	|:------:	|:------:	|
| document 1 	|    1   	|    1   	|    1   	|    1   	|
| document 2 	|    4   	|    5   	|    0   	|    0   	|
| document 3 	|    0   	|    0   	|    1   	|    1   	|
| Query      	|    1   	|    1   	|    0   	|    0   	|


## Distributional Semantic Models

The linguistic theory of distributional semantics is based on the hypothesis, that words, which occur in similar contexts, have similar meaning. J.R. Firth formulated this assumption in his famous sentence {cite}`firth57synopsis`: 

*You shall know a word by the company it keeps*


Since computers can easily determine the co-occurrence statistics of words in large corpora the theory of distributional semantics provides a promising opportunity to automatically learn semantic relations. The learned semantic representations are called **Distributional Semantic models (DSM)**. They represent each word as a numerical vector, such that words, which appear frequently in similar contexts, are represented by similar vectors. In the figure below the arrow represents the DSM. Since there are many different DSMs there a many different approaches to implement this transformation from the hypothesis of distributional semantics to the word space. 



<figure align="center">
<img width="500" src="https://maucher.home.hdm-stuttgart.de/Pics/semanticsInWordSpace.png">
<figcaption>Mapping the hypothesis of distributional semantics to a word space</figcaption>
</figure>


The field of DSMs can be categorized into the classes **count-based models** and **prediction-based models**. Recently, considerable attention has been focused on the question which of these classes is superior {cite}`pennington2014`.


### Count-based DSM

### Prediction-based DSM 

