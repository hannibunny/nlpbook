# Vector Representations of Words and Documents

As described in the {doc}`../intro`, many NLP tasks can either be solved in a rule-based or in a data-based approach. Data-based approaches increasingly yield better results than rule-based approaches. At the heart of data-based approaches is a Machine Learning algorithm, which learns a model from available training data. This model can then be applied e.g. for Document Classification, Sentiment Analysis, Named-Entity-Recognition, Intent-Recognition, Automatic translation and many other NLP tasks.

All ML-algorithms require a numeric representation at their input, usually a fixed-length numeric vector. In the context of NLP the input is usually text. The crucial question is then:

```{admonition} Question:
How to represent texts, i.e. sequences of words, punctuation marks..., as a numeric vector of constant length?
```

In this section first the general notion of **Vector Space Model** is introduced. Then the most common type of vector space model for text, the so called **Bag-Of-Word (PoW)** model is described. In the extreme case of single-word-texts the BoW melts down to the **One-Hot-Encoding** of words. The pros and cons of these conventional representations are discussed. Another method to represent words as numerical vectors is constituted by **Distributional Semantic Models (DSMs)**. The currently very popular **Word Embeddings** belong to the class of DSMs. **Word Embeddings** are numerical word representations, which are learned by *Neural Networks*. Even though they have been considered before, the 2013 milestone paper {cite}`NIPS2013_5021` introduced two very efficient methods to learn meaningful Word Embeddings. Since then Word Embeddings have revolutionized NLP.

## Vector Space Model

Vector Space Models map arbitrary inputs to numeric vectors of fixed length. For a given task, you are free to define a set of $N$ relevant features, which can be extracted from the input. Each of the $N$-feature extraction functions returns how often the corresponding feature appears in the input. Each component of the vector-representation belongs to one feature and the value at this component is the count of this feature in the input.

```{admonition} Example: Vector Space Model in General
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

Numeric vector presentation of documents are not only required for Machine-Learning based NLP tasks. Another important application category is **Information Retrieval (IR)**. Information Retrieval deals with algorithms and models for searching information in large document collections. Web-search like [www.google.com](www.google.com) is only one example for IR. In such document-search applications the user defines a **query**, usually in terms of one or more words. The task of the IR system is then to
 
* return relevant documents, which match the query
* rank the returned documents, such that the most important is at the top of the search-result

Challenges in this context are:

* How to deduce what the user actually wants, given only a few query-words?
* How to calculate the **relevance** of a document with respect to the query words?

Question 1 will be addressed later further below, when *Distributional Semantic Models*, in particular *Word Embeddings* are introduced. The second question will be answered now.

The conventional approach for document search is to 

1. model all documents in the index as numerical vectors, e.g. by BoW 
2. model the query as a numerical vector in the same way as the documents are modelled
3. determine the most relevant documents by just determining the document-vectors, which have the smallest distance to the query-vector.

This means that the question on *relevance* is solved by determining *nearest vectors in a vector space*. 
An example is given below. Here we assume, that there are only 3 documents in the index and there are only 4 different words, occuring in these documents. Document 2, for example, contains *word 1* with a frequency of 4 and *word 2* with a frequency of 5. The query consists of *word 1* and *word 2*. The BoW-matrix and the attached query-vector are:    


|            	| word 1 	| word 2 	| word 3 	| sord 4 	|
|------------	|:------:	|:------:	|:------:	|:------:	|
| document 1 	|    1   	|    1   	|    1   	|    1   	|
| document 2 	|    4   	|    5   	|    0   	|    0   	|
| document 3 	|    0   	|    0   	|    1   	|    1   	|
| Query      	|    1   	|    1   	|    0   	|    0   	|

Given these vector-representations, it is easy to determine the distances between each document and the query. 

The obvious *type of distance* is the **Euclidean Distance**: For two vectors, $\underline{a}=(a_1,\ldots,a_n)$ and $\underline{b}=(b_1,\ldots,b_n)$, the Euclidean Distance is defined to be

$$
d_E(\underline{a},\underline{b})=\sqrt{\sum_{i=1}^n (a_i-b_i)^2}
$$

**Similarity** and **Distance** are inverse to each other, i.e. the similarity between vectors increases with decreasing distance and vice versa. For each distance-measure a corresponding similarity-measure can be defined. E.g. the *Euclidean-distance*-based similarity measure is 

$$
s_E(\underline{a},\underline{b})=\frac{1}{1+d_E(\underline{a},\underline{b})}
$$

Now let's determine the Euclidean distance between the query and the 3 documents in the example abover:

Euclidean distance between query and document 1:

$$
d_E(\underline{q},\underline{d}_1)=\sqrt{(1-1)^2+(1-1)^2+(1-0)^2+(1-0)^2} = \sqrt{2} = 1.41
$$

Euclidean distance between query and document 2:

$$
d_E(\underline{q},\underline{d}_2)=\sqrt{(4-1)^2+(5-1)^2+(0-0)^2+(0-0)^2} = \sqrt{25} = 5.00
$$

Euclidean distance between query and document 3:

$$
d_E(\underline{q},\underline{d}_3)=\sqrt{(0-1)^2+(0-1)^2+(1-0)^2+(1-0)^2} = \sqrt{4} = 2.00
$$

Comparing these 3 distances, one can conclude, that document 1 has the smallest distance (and the highest similarity) to the query and is therefore the best match. 

*Is this what we expect?* 

*No!* Document 2 contains the query words not only once but with a much higher frequency. One would expect, that this stronger prevalence of the query words implies that Document 2 is more relevant. 

*So what went wrong?*

The answer is, that **the Euclidean Distance is just the wrong distance-measure** for this type of application. In a query each word is contained only once. Therefore, Euclidean-distance penalizes longer documents with more words. 

The solution to this problem is 

* either normalize all vectors - document vectors and query-vector - to unique length,
* or apply another distance measure

The standard similarity-measure for BoW vectors is the **Cosine Similarity**, which is calculated as defined in the table below. The table also contains the definition of the corresponding distance-measure. Moreover, a bunch of other distance- and similarity measures, which are frequently applied in NLP tasks, are listed in the table.  


<figure align="center">
<img width="700" src="https://maucher.home.hdm-stuttgart.de/Pics/distanceMeasures.png">
</figure> 


For the query-example above, the Cosine-Similarities are:

Cosine Similarity between query and document 2:

$$
s_C(\underline{q},\underline{d}_1)=\frac{1 \cdot 1 + 1 \cdot 1 + 1 \cdot 0 + 1 \cdot 0}{\sqrt{4} \cdot \sqrt{2}} = \frac{1}{\sqrt{2}} = 0.707
$$

Cosine Similarity between query and document 2:

$$
s_C(\underline{q},\underline{d}_2)=\frac{4 \cdot 1 + 5 \cdot 1 + 0 \cdot 0 + 0 \cdot 0}{\sqrt{41} \cdot \sqrt{2}} = \frac{9}{\sqrt{82}} = 0.994
$$

Cosine Similarity between query and document 3:

$$
s_C(\underline{q},\underline{d}_3)=\frac{0 \cdot 1 + 0 \cdot 1 + 1 \cdot 0 + 1 \cdot 0}{\sqrt{2} \cdot \sqrt{2}} = \frac{0}{2} = 0
$$

These calculated similarities match our subjective expectation: The similarity between document 3 and query q is 0 (the lowest possible value), since they have no word in common. The similarity between document 2 and the query q is close to the maximum similarity-value of 1, since both query-words appear with a high frequency in this document.

### BoW Drawbacks

BoW representation of documents and the One-Hot-Encoding of single words, as described above, are methods to map words and documents to numeric vectors, which can be applied as input for arbitrary Machine Learning algorithms. Hovever, these representations suffer from crucial drawbacks: 

1. The vectors are usually very long - there length is given by the number of words in the vocabulary. Moreover, the vectors are quite sparse, since the set of words appearing in one document is usually only a very small part of the set of all words in the vocabulary.
2. Semantic relations between words are not modelled. This means that in this model there is no information about the fact that word *car* is more related to word *vehicle* than to word *lake*. 
3. In the BoW-model of documents word order is totally ignored. E.g. the model can not distinguish if word *not* appeared immediately before word *good* or before word *bad*.  

All of these drawbacks can be solved by applying *Distributional Semantic models* to map words into numeric vectors and by the way the resulting *Word Empeddings* are passed e.g. to the input of Recurrent Neural Networks, Convolutional Neural Networks or Transformers (see later chapters of this lecture). 


## Distributional Semantic Models

The linguistic theory of distributional semantics is based on the hypothesis, that words, which occur in similar contexts, have similar meaning. J.R. Firth formulated this assumption in his famous sentence {cite}`firth57synopsis`: 

*You shall know a word by the company it keeps*


Since computers can easily determine the co-occurrence statistics of words in large corpora the theory of distributional semantics provides a promising opportunity to automatically learn semantic relations. The learned semantic representations are called **Distributional Semantic models (DSM)**. **They represent each word as a numerical vector, such that words, which appear frequently in similar contexts, are represented by similar vectors.** In the figure below the arrow represents the DSM. Since there are many different DSMs there a many different approaches to implement this transformation from the hypothesis of distributional semantics to the word space. 


<figure align="center">
<img width="500" src="https://maucher.home.hdm-stuttgart.de/Pics/semanticsInWordSpace.png">
<figcaption>Mapping the hypothesis of distributional semantics to a word space</figcaption>
</figure>


The field of DSMs can be categorized into the classes **count-based models** and **prediction-based models**. Recently, considerable attention has been focused on the question which of these classes is superior {cite}`pennington2014`.


### Count-based DSM

DSMs map words to numeric vectors, such that semantically related words, i.e. words which appear frequently in a similar context, have similar numeric vectors. The first question which arises from this definition is *What is context?* In all DSMs, introduced in this lecture, the context of a target word $w$ is considered to be the sequence of $L$ previous and $L$ following words, where the *context-length* $L$ is a parameter.

I.e. in the word-sequence

$$
\ldots,w_{i-L},\ldots,w_{i-1},w_i,w_{i+1},\ldots,w_{i+L},\ldots
$$

the words $w_{i-L},\ldots,w_{i-1},w_{i+1},\ldots,w_{i+L}$ constitute the context of target word $w_i$ and 

$$
\left\{(w_i,w_{i+j})\right\}, \, j \in \{-L,\ldots,-1,1,\ldots,L\}
$$

is the corresponding *set of word-context-pairs* w.r.t. target word $w_i$.

The most common numeric vector representation of count-based DSMs can be derived from the **Word-Co-Occurence** matrix. In this matrix each row belongs to a target-word and each column belongs to a context word. Hence, the matrix has usually $\mid V \mid$ rows and the same amount of columns[^F1]. In this matrix the entry in row $i$, column $j$ is the number of times context word $c_j$ appears in the context of target word $w_i$. This frequency is denoted by $\#(w_i,c_j)$. The structure of such a word-co-occurence matrix is given below: 

[^F1]: One may have different vocabularies for target-words ($V_W$) and context-words ($V_C$). However, often they coincide, i.e. $V = V_W=V_C$.


<figure align="center">
<img width="600" src="https://maucher.home.hdm-stuttgart.de/Pics/cooccurenceMatrix.png">
<figcaption>Word-Cooccurence-Matrix</figcaption>
</figure>

In order to determine this matrix a large corpus of contigous text is required. Then for all target-context pairs the corresponding count $\#(w_i,c_j)$ must be determined. **Once the matrix is complete the numeric vector representation of word $w_i$ is just the i.th row in this matrix!**.


```{admonition} Example: Word-Co-Occurence Matrix
:class: dropdown

Assume that the (unrealistically small) corpus is 

K = [The dusty road ends nowhere. The dusty track ends there.*

For a (unrealistically small) context length of $L=2$ the word-co-occurence matrix is:

<figure align="center">
<img width="600" src="https://maucher.home.hdm-stuttgart.de/Pics/cooccurenceMatrixExample.png">
<figcaption>Example of Word-Cooccurence-Matrix</figcaption>
</figure>

With this matrix, the numeric vector representation of word **road** is:

$$
(1,1,0,1,1,0,0)
$$ 

and the vector for word **track** is:

$$
(1,1,0,1,0,0,1)
$$

As can be seen, these two vectors are quite similar. The reason for this similarity is that both words appear in similar contexts. Hence we assume that they are semantically correlated.

```

### Prediction-based DSM 

