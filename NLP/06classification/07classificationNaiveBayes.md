# Naive Bayes Text Classification

Classifiers must determine the most likely class $C_i$ for a given input $\underline{x}$. Using probability theory, this task can be described as follows: For all possible classes $C_i$ determine the conditional probability 

$$
P(C_i | \underline{x}),
$$ 

i.e. the probability that the given input $\underline{x}$ falls into class $C_i$. Then select the class for which this probability is maximum.

In the training phase the Naive Bayes Algorithm estimates the so-called **likelihood function** 

$$
p(\underline{x} \mid C_i)
$$ 

and the **a-priori** class probabilities

$$
P(C_i)
$$

from the given training data. In the inference phase it then applies the **Bayes-Rule** for determining the **a-posteriori**-probability

$$
P(C_i \mid \underline{x}) = \frac{p(\underline{x} \mid C_i) \cdot P(C_i)}{p(\underline{x})}.
$$ (bayes-rule)

Then the class $C_i$, for which $P(C_i \mid \underline{x})$ is maximum is determined. Note that for determining the class, for which $P(C_i \mid \underline{x})$ is maximal, doesn't require the so-called *evidence* $p(\underline{x})$, since this value is independent of the class, i.e.

$$
argmax_i(P(C_i \mid \underline{x})) = argmax_i (\frac{p(\underline{x} \mid C_i) \cdot P(C_i)}{p(\underline{x})})  = argmax_i (p(\underline{x} \mid C_i) \cdot P(C_i))
$$ (bayes-inference)

The calculation of the likelihood function $p(\underline{x} \mid C_i)$ is quite difficult if $\underline{x}=(x_1,x_2,\ldots,x_Z)$ is a vector of interdependent variables $x_i$. However, the **Naive Bayes Classification** is based on the simplifying assumption that the input variables $x_i$ are independent of each other. In this case, the conditional compound probability function $p(x_1,x_2,\ldots,x_Z \mid C_i)$ is simplified to

$$
p(x_1,x_2,\ldots,x_Z \mid C_i)=\prod\limits_{j=1}^Z p(x_j | C_i).
$$ (bayes-assumption)

Hence, in the inference face the classifier must determine

$$
argmax_i \left( \prod\limits_{j=1}^Z p(x_j | C_i) \cdot P(C_i) \right)
$$ (bayes-inference-general)

The conditional probability $p(x_j | C_i)$ is the probability, that in documents of class $C_i$ the value of the $j.th$ component in the feature-vector is $x_j$. 

**If the Naive Bayes Classifier is applied for text-classification**, then $p(x_j | C_i)$ describes the probability, that word $x_j$ appears at least once in a document of class $C_i$. This probability can be estimated by

$$
p(x_j | C_i) = \frac{\#(x_j,C_i)}{\#(C_i)},
$$ (bayes-estimate-conditional)

where $\#(x_j,C_i)$ is the number of class $C_i$-training-documents, which contain word $x_j$ and $\#(C_i)$ is the number of class $C_i$-training-documents.

The a-priori-possiblities are estimated by

$$
P(C_i)=\frac{\#(C_i)}{N},
$$ (bayes-estimate-apriori)

where $N$ is the total number of training-documents.

Moreover, inference, in the case that the Naive Bayes algorithm is applied for text-classification applies the following variant of equation {eq}`bayes-inference-general`:

$$
argmax_i \left( \prod\limits_{x_j \in D} p(x_j | C_i) \cdot P(C_i) \right),
$$ (bayes-inference-text)

where $D$ is the set of all words, which are contained in the document that shall be classified. In contrast to equation {eq}`bayes-inference-general`, in equation {eq}`bayes-inference-text` the product is calculated only over the words, contained in the current document.

(exampleNB)=
```{admonition} Example: Naive Bayes Spam Filter

The following labeled Emails are available for training a Naive Bayes Classifier:

* 4 Training documents labeled with class `Good`:

		- nobody owns the water
		- the quick rabbit jumps fences
		- the quick brown fox jumps
		- next meeting is at night
		
* 4 Training documents labeled with class `Bad`:

		- buy pharmaceuticals now
		- make quick money at the online casino
		- meeting with your superstar
		- money like water
		
**Task:** Determine the class (Bad or Good) which is assigned to the new Email

		- the money jumps

by a Naive Bayes Classifier.

**Solution:**

1. Determine for the relevant words `the`, `money` and `jumps` and both classes the conditional probabilites according to equation {eq}`bayes-estimate-conditional`:
 
     \begin{eqnarray*}
 		P(the|Bad) & = & 0.25 \\
 		P(the|Good) & = & 0.75 \\
 		P(money|Bad) & = & 0.5  \\
 		P(money|Good) & = & 0.0 \\
 		P(jumps|Bad) & = & 0.0  \\
 		P(jumps|Good) & = & 0.5 \\
 	\end{eqnarray*}
 
2. Determine for both classes the a-priori probability according to equation {eq}`bayes-estimate-apriori`:
 
    \begin{eqnarray*}
		P(Bad) & = & 0.5 \\
		P(Good) & = & 0.5 \\
	\end{eqnarray*} 

3. Determine the class, for which the argument of equation {eq}`bayes-inference-text` is maximal:

    \begin{eqnarray*}
		\mbox{Class Bad:  }  & 0.25 \cdot 0.5 \cdot 0.0 \cdot 0.5 = 0 \\
		\mbox{Class Good:  } & 0.75 \cdot 0.0 \cdot 0.5 \cdot 0.5 = 0 \\
    \end{eqnarray*}

For both classes the same a-posteriori probability value has been calculated. In this case no classification is possible!
	
```

The example above uncovers a drawback of Naive Bayes classification. If the document, that shall be classified, contains a word $x_j$, which does not appear in the class $C_i$-training data, the corresponding conditional probability $P(x_j|C_i)$ is zero and as soon as one of the factors of {eq}`bayes-inference-text` is zero, the entire product is zero. In order to avoid this problem of zero-factors, Naive Bayes classifiers are usually modified by applying <font color="red"> smoothing </font>.

Smoothing in the context of Naive Bayes classification means, that the conditional probabilities are estimated not as defined in equation {eq}`bayes-estimate-conditional`, but in a slightly modified form, which guarantees that the values of the smoothed conditional probabilities are always non-zero. There exist different smoothing techniques. An approach, which is frequently applied for smoothing in the context of Naive Bayes document classification is defined by replacing the conditional probabilities $p(x_j|C_i)$ 
in equation {eq}`bayes-inference-text` by the following weighted conditional probabilities:


$$
	p_{weight}(x_j \mid C_i)=\frac{w \cdot P_{ass,i,j} + |x_j| \cdot P(x_j \mid C_i)}{w+|x_j|},
$$ (nb-smoothing)

where

 * $P_{ass,i,j}$ is an assumed probability that a word $x_j$ belongs to class $C_i$. By default this probability can be set to $1/K$, where $K$ is the number of classes, that must be distinguished. However, this probability can also be set individually, e.g. for word *viagra* the probabilities may be $P_{ass,viagra,bad}=0.95$ and $P_{ass,viagra,good}=0.05$. I.e. this term provides the possibility to integrate prior-knowledge.  
 * $w$ is a weight-factor, which can be set in order to control how strong the assumed probability $P_{ass,i,j}$ contributes to the smoothed probability $p_{weight}(x_j \mid C_i)$. Default: $w=1$.
 
 For the classification of a new document containing words $D$, instead of equation {eq}`bayes-inference-text`, the following equation is applied: 
  
 $$
     argmax_i \left( \prod\limits_{x_j \in D} p_{weight}(x_j | C_i) \cdot P(C_i) \right),
 $$ (bayes-inference-text-smoothed)
 
 ```{admonition} Example: Naive Bayes Spam Filter with Smooting
 
 The given training data is the same as in example {ref}`exampleNB`. The smoothed conditional probabilities according to {eq}`nb-smoothing` are then:
 
    \begin{eqnarray*}
      P_{weight}(the|Bad) & = & \frac{0.5 + 4 \cdot 0.25}{1+4}=0.3 \\
      P_{weight}(the|Good) & = & \frac{0.5 + 4 \cdot 0.75}{1+4}=0.7 \\
      P_{weight}(money|Bad) & = & \frac{0.5 + 2 \cdot 0.5}{1+2}=0.5 \\
      P_{weight}(money|Good) & = & \frac{0.5 + 2 \cdot 0.0}{1+2}=0.167 \\
      P_{weight}(jumps|Bad) & = & \frac{0.5 + 2 \cdot 0.0}{1+2}=0.167 \\
      P_{weight}(jumps|Good) & = & \frac{0.5 + 2 \cdot 0.5}{1+2}=0.5 \\
     \end{eqnarray*}
	
The a-priori class probabilities are unchanged:

    \begin{eqnarray*}
		\mbox{Class Bad:  }  & 0.25 \cdot 0.5 \cdot 0.0 \cdot 0.5 = 0 \\
		\mbox{Class Good:  } & 0.75 \cdot 0.0 \cdot 0.5 \cdot 0.5 = 0 \\
    \end{eqnarray*}
	
Applying the smoothed probabilities, the class, for which the argument of equation {eq}`bayes-inference-text-smoothed` is maximal for the new email $(the, money, jumps)$:

	\begin{eqnarray*}
		\mbox{Class Bad:  }   & 0.3 \cdot 0.5 \cdot 0.167 \cdot 0.5 = 0.0125 \\
		\mbox{Class Good:  }  & 0.7 \cdot 0.167 \cdot 0.5 \cdot 0.5 = 0.029 \\
	\end{eqnarray*}
	
Hence this email is classified to be `Good`.
	
 
 ```
 
 
 
 Note, that for classification it was sufficient to calculate 
 
 $$
  p(\underline{x} \mid C_i) \cdot P(C_i),
 $$ 
 
 for all classes $C_i$ and decide on the class, for which this value is maximal. However, the calculated values are not the a-posteriori-probabilities $P(C_i | \underline{x})$, since we ignored the evidence $p(\underline{x})$ (see equation {eq}`bayes-inference`).
 However, we can easily obtain the a-posteriori-probabilities $P(C_i | \underline{x})$, by applying the marginalisation-rule for the calculation of the evidence in the denominator of the Bayes Rule (equation {eq}`bayes-rule`) 
 
 $$
    p(\underline{x}) = \sum\limits_{i=1}^K p(\underline{x},C_i) = \sum\limits_{i=1}^K p(\underline{x} | C_i) \cdot P(C_i).
 $$
 
 In the example above, the a-posteriori probabilities are then:
 
 \begin{eqnarray*}
  P(Bad | (the,money,jumps)) & = & \frac{0.0125}{0.0125+0.029} & = & 0.30 \\
  P(Good | (the,money,jumps)) & = & \frac{0.029}{0.0125+0.029} & = & 0.70 \\   

 \end{eqnarray*}
 
 
