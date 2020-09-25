## N-Gram Language Model

An **N-gram** is a sequence of N consecutive words. For example from the text *the traffic lights switched from green to yellow*, the following set of 3-grams (N=3) can be extracted:

* (the, traffic, lights)
* (traffic, lights, switched)
* (lights, switched, from)
* (switched, from, green)
* (from, green, to)
* (green, to, yellow)

### Generating a probabilistic language model

N-grams can be applied to create a **probabilistic language model** (also called N-gram language model). For this a large corpus of consecutive text(s) is required. *Consecutive* means that the order of words and sentences is kept like in the original document. The corpus need not be annotated. Hence, one can apply for example an arbitrary book, or a set of books, to learn a language model. For this the frequency of all possible N-grams and (N-1)-grams must be determined. From these frequencies the conditional probabilities 

$$
P(w_n|w_1,w_2,w_3,\ldots,w_{n-1})
$$   

that word $w_n$ follows the words $w_1,w_2,w_3,\ldots,w_{n-1}$ can be estimated as follows:


$$
P(w_n|w_1,w_2,w_3,\ldots,w_{n-1}) = \frac{\#(w_1,w_2,w_3,\ldots,w_{n-1},w_n)}{\#(w_1,w_2,w_3,\ldots,w_{n-1})}
\label{eq:condprobest} \tag{1}
$$

where $\#(w_1,w_2,w_3,\ldots,w_{n-1},w_n)$ is the frequency of N-gram $(w_1,w_2,w_3,\ldots,w_{n-1},w_n)$ and $\#(w_1,w_2,w_3,\ldots,w_{n-1})$ is the frequency of N-gram $(w_1,w_2,w_3,\ldots,w_{n-1})$.

### Applications of language models

The possibility to estimate the likelihood of words, given the $(N-1)$ previous words allows application such as,

* detection of erroneous words in text. An erroneous word usually has a conditional probabiliy close to zero.
* correction of erroneous words: Replace the very unlikely word (error) by the most likely one for the given predecessors
* the determinantion of the most likely successor for given $(N-1)$ predecessors enables text-completion applications, such as the ones in the figures below.  
* the determinantion of the most likely successors is a crucial factor for automatic text generation and automatic translation.

<figure align="center">
<img width="250" src="https://maucher.home.hdm-stuttgart.de/Pics/quicktype.PNG">
<figcaption><b>Figure:</b> Word Completion in Messenger</figcaption>
</figure>

<figure align="center">
<img width="400" src="https://maucher.home.hdm-stuttgart.de/Pics/wordproposal.PNG">
<figcaption><b>Figure:</b>Sueggestions for refining queries in web-search</figcaption>
</figure>

### The power of context

Try to decode the text in the figure below:

<figure align="center">
<img width="400" src="https://maucher.home.hdm-stuttgart.de/Pics/KontextBeimLesen.jpeg">
<figcaption><b>Figure:</b>The capability to infer from context enables us to decode strongly corrupted text</figcaption>
</figure>

Did you get it? If so, the reason for your success is that humans exploit context to infer missing knowledge.

In the example above context is given by surrounding characters, word-length, similarity of characters etc. Applications of N-gram language models understand context to be the $(N-1)$ previous words. The obvious question is then *What is a suitable value for N?*. Certainly, the larger $N$, the more context is integrated and the larger the knowledge. However, with an increasing $N$ the probability that all N-grams appear often enough in the training-corpus, such that the conditional probabilities are robust, decreases. Moreover, the memory required to save the probabilistic model increases.

### Probabilities for arbitrary word-sequences

Given the conditional probabilities of the language model, the *joint probability* $P(x_1 \ldots, x_Z)$ for a wordsequence of arbitrary length $Z$ can be calculated.

For two random variables $x$ and $y$ the relation between the *joint probability* and the *conditional probability* is:

$$
	P(x,y)=P(x\mid y) \cdot P(y)
$$

The generalisation of this relation to an arbitrary amount of $Z$ random variables  $x_1 \ldots, x_Z$ is the **Chain Rule**:

$$
P(x_1 \ldots, x_Z) \\ 
= P(x_Z \mid x_1 \ldots, x_{Z-1}) \cdot P(x_1 \ldots, x_{Z-1}) \\
= P(x_Z \mid x_1 \ldots, x_{Z-1}) \cdot P(x_{Z-1} \mid x_1 \ldots, x_{Z-2}) \cdot P(x_1 \ldots, x_{Z-2}) \\
= P(x_Z \mid x_1 \ldots, x_{Z-1}) \cdot P(x_{Z-1} \mid x_1 \ldots, x_{Z-2}) \cdot \ldots \cdot P(x_2 \mid x_1) \cdot P(x_1) 
$$
	

In the case of an N-gram language model and $Z>N$ this chain rule becomes much simpler, because the N-Gram model assumes, that each word depends only on it's $(N-1)$ predecessors. If this assumption is true, than

$$
P(x_Z \mid x_1 \ldots, x_{Z-1}) = P(x_Z \mid x_{Z-N+1} \ldots, x_{Z-1}).
$$
For $Z>N$ the term on the right hand side is simpler than the term on the left, since only a limited part of the *history* must be regarded. Hence the chain rule gets simpler in the sense, that the condition in the conditional probabilities consists of less elements. Hence, the probability for a wordsequence of arbitrary length $Z$ can be calculated as follows:


$$
P(x_1 \ldots, x_Z) =
$$ 
$$
P(x_Z \mid x_{Z-N+1} \ldots, x_{Z-1}) \cdot P(x_{Z-1} \mid x_{Z-N} \ldots, x_{Z-2}) \cdot \ldots \cdot P(x_2 \mid x_1) \cdot P(x_1) \label{eq:chainruleN} \tag{2}
$$


```{admonition} Example
:class: dropdown
According to the chain-rule, the probability for the word-sequence `the traffic lights switched from green to yellow` is

$$
P(the, traffic, lights, switched, from, green, to, yellow) = \\ 
P(yellow \mid the, traffic, lights, switched, from, green, to) \cdot \\ 
P(to \mid the, traffic, lights, switched, from, green) \cdot \\
P(green \mid the, traffic, lights, switched, from) \cdot \\
P(from \mid the, traffic, lights, switched) \cdot \\
P(switched \mid the, traffic, lights) \cdot \\
P(lights \mid the, traffic) \cdot \\ 
P(traffic \mid the) \cdot \\ 
P(the) 
$$  

However, if a 3-Gram language model is assumed, the required conditional probability factors become much simpler:

$$
P(the, traffic, lights, switched, from, green, to, yellow) = \\
P(yellow \mid green, to) \cdot \\
P(to \mid from, green) \cdot \\
P(green \mid switched, from) \cdot \\
P(from \mid lights, switched) \cdot \\
P(switched \mid traffic, lights) \cdot \\
P(lights \mid the, traffic) \cdot \\
P(traffic \mid the) \cdot \\
P(the)
$$ 
```

### Estimating the Probabilities of an N-Gram Language Model

#### Maximum Likelihood Estimation 

A trained N-gram language model consists of conditional probabilities, determined from the given training corpus. These probabilities can be estimated as defined in equation $\eqref{eq:condprobest}$. This way of estimating the probabilities is called **Maximum Likelihood Estimation**. Even though this method of estimation sounds obvious, it has a significant drawback, which makes it impossible for practical applications: As soon as there is an N-gram in the application-text, which is not contained in the training-corpus, the corresponding conditional probability is 0. If only one factor in equation $\eqref{eq:chainruleN}$ is zero, the entire product and thus the probability for the word-sequence is also zero, independent of the values of the other factors in the product. In order to avoid this, different **smoothing**-techniques have been developed. Two of them, *Laplace-Smoothing* and *Good-Turing-Smoothing* are described in the following subsections.

#### Laplace Smoothing

A trivial method to avoid that conditional probabilities, which are calculated as in equation $\eqref{eq:condprobest}$ is to just $add 1$ to the nominator. I.e. instead of $\#(w_1,w_2,w_3,\ldots,w_{n-1},w_n)$ we use $\#(w_1,w_2,w_3,\ldots,w_{n-1},w_n)+1$ in the nominator. By adding such a bias of 1, we implictly assume, that each possible $n-gram$ appears one times more than it's actual frequency in the corpus. With this assumption, how much more $n-grams$ with the same first $(n-1)$ words do we then have in this *virtually extended corpus*? Since each word of the vocabulary, can be the last word of a $n$-gram with fixed subsequence $(w_1,w_2,w_3,\ldots,w_{n-1})$, the answer is $\mid V \mid$, the number of different words in the vocabulary. This value must be added to the denominator for calculating the **Laplace-Smoothed conditional Probability**:
 
$$
P_{L}(w_n|w_1,w_2,w_3,\ldots,w_{n-1}) = \frac{\#(w_1,w_2,w_3,\ldots,w_{n-1},w_n)+1}{\#(w_1,w_2,w_3,\ldots,w_{n-1})+\mid V \mid}
\label{eq:laplacesmooth} \tag{3}
$$

By applying in equation $\eqref{eq:chainruleN}$ the Laplace-smoothed conditional probabilities of equation $\eqref{eq:laplacesmooth}$ instead of the *true* conditional probabilities of equation $\eqref{eq:condprobest}$, the probability of a word-sequence can never be zero.

Laplace smoothing is in general the most common smoothing technique, to avoid zero-factors in probability-calculations like this. However, in the context of N-gram language model Laplace smoothing is not advisable, because it distorts the *true* conditional probabilities too much. For example, assume that we like to calculate the conditional probability $P(to \mid want)$ that `to` follows `want` for a Bigram (N=2) language model. Assume that the Unigram (N=1) $(want)$ appears 927 times and the bigram $(want, to)$ appears 608 times in the corpus. Moreover, we assume a relatively small vocabulary of only $\mid V \mid = 1446$ different words. Then the *true* conditional probability according to Maximum-Likelihood-Estimation is 

$$
P(to \mid want) = \frac{608}{927} = 0.66
$$

However, with Laplace-Smoothing, the conditional probability is

$$
P_{L}(to \mid want) = \frac{608+1}{927+1446} = 0.26
$$

By comparing the two values, the immense distortion becomes obvious. The distortion increases with the value, which must be added in the denominator. In the case of N-Gram language models this value is the number of different words in the vocabulary, which is usually quite high. A less distorting smoothing-technique for language models is *Good-Turing-Smoothing*. 

#### Good-Turing-Smoothing

Assume that we have a box, which contains balls (*objects*) of different color (*species*). We do not know how many balls are in the box and we also do not know the number of different colors. After drawing

* 10 red balls, 
* 3 green balls, 
* 2 blue balls 
* 1 black ball
* 1 brown ball
* 1 grey ball

we ask for **the probability that the next ball we draw has a specific color**. For the previously seen colors, the probability can be estimated by Maximum-Likelihood. E.g. the probability, that the next ball is black is

$$
P(black)= \frac{1}{18}.
$$


More interesting is the question:

* **What is the Probability, that the next ball (*object*) has an so far unseen color (*species*)?**

Good-Turing's assumption to answer this question is:

* **The frequency of so far unseen species is the same as the frequency of the species, which have been observed only once, so far**

In the example above the frequency of colors, which have been seen only once so far is 3. Hence, the probability that the next drawn ball has a so far unseen color is

$$
P(unseen)=\frac{3}{18}
$$

For the general definition we use the following notation:

* $R_x$ indicates how often species $x$ has been observed so far
* $N_r$ indicates the the number of species, which has been seen $r$ times so far
* $Z$ is the number of total observations so far

With this notation, we can reformulate the Good-Turing assumption to be $N_0 := N_1$. Moreover, the number of total observations so far, can be calculated as follows:

$$
Z = \sum\limits_{r=1}^{\infty}N_r \cdot r.
\label{eq:sumGT} \tag{4}
$$ 

The probability that the next drawn *object* has a so far unseen *species* is

$$
P_{GT}(unseen)=\frac{N_1}{Z}
\label{eq:punseen} \tag{5}
$$

and if we know the number of so far unseen species $N_0$, than also the probability for a concrete so far unseen species can be calculated to be

$$
P_{GT}(x)=\frac{N_1}{N_0 \cdot Z},
\label{eq:punsingle} \tag{6}
$$  

where $x$ is a so far unseen species. With respect to the example above, if we know that *yellow* and *orange* are the only so far unseen colors (*species*), then the probability that the next drawn ball is orange would be

$$
P(x)=\frac{3}{2 \cdot 18} = \frac{1}{12}
$$

If we stop here, the resulting probabilities would not be valid, because a fundamental law of probability theory would be violated, which says, that the sum over all probabilities must be 1:

$$
\sum\limits_{x \in D} P(x) =1,
$$
where $D$ is the domain of possible values for $x$. This law would be violated, since we $virtually$ added observations, which have actually not been occured (we assumed that the unseen events occured as often as the events, which have been seen once). 

Good-Turing-Smoothing solves this dilemma by adapting the frequency values $r$. It is pretended, that species, which actually occured $r$ times, occured $r*$ times, with

$$
r* = \frac{(r+1)N_{r+1}}{N_r}.
\label{eq:rstern} \tag{7}
$$

With this adaptation the total sum of observations remains the same:

$$
\sum\limits_{r=0}^{\infty}N_{r} \cdot \frac{(r+1)N_{r+1}}{N_r} = \sum\limits_{r=0}^{\infty} (r+1)N_{r+1} = \sum\limits_{r=1}^{\infty}N_r \cdot r = Z.
$$  

The Good-Turing-smoothed probability of a species $x$, which actually has been observed $r$ times is 

$$
P_{GT}(x) = \frac{r*}{Z}
$$

 

In our example of drawing balls of different colors, the Good-Turing smoothded probability for drawing a black ball is then 

$$
P_{GT}(black) = \frac{2 \cdot \frac{1}{3}}{18} = \frac{2}{3 \cdot 18},
$$
 
since for $r=1$ (black ball appeared once so far), we have

$$
r* = \frac{(r+1)N_{r+1}}{N_r} = \frac{2 \cdot 1}{3}.
$$


Note that the calculation of $r*$ according to equation $\eqref{eq:rstern}$ fails, if $N_{r+1}=0$. Therefore, it is suggested to apply expectation values $E(N_{r+1})$ and $E(N_{r})$ in equation $\eqref{eq:rstern}$. Such expectation values can be determined, e.g. by interpolation.

**Final remark on the number of unseen events in N-gram language models:** From the corpus, the frequency $R_x$ for all N-grams in the corpus can the determined. Then also all $N_r$-values (number of N-grams which appear r times) can be determined. For an N-gram $x$, which is not in the corpus, we like to determine the probability $P_{GT}(x)$ according to equation $\eqref{eq:punsingle}$. But what is $N_0$, the number of unseen N-grams? For this we first determine the number $Z$ of observed N-grams according to $\eqref{eq:sumGT}$. Then we subtract Z from the total number of possible N-grams:

$$
N_0 = \mid V \mid^{N} - \sum\limits_{r=1}^{\infty}N_r \cdot r .
$$ 



 


