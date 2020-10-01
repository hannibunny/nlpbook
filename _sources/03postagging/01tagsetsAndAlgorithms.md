## PoS Tagsets

Depending on the language and the NLP task different tagsets can be applied. Popular English and German tagsets are:

* [Penn Treebank Tagset](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)
* [Tagset of Brown Corpus](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)
* [Tagset of the British National Corpus](http://www.natcorp.ox.ac.uk/docs/c5spec.html)
* [Stuttgart-Tübingen-Tagset](https://www.ims.uni-stuttgart.de/forschung/ressourcen/lexika/germantagsets/#id-cfcbf0a7-0)

In NLP tools (e.g. [NLTK](https://www.nltk.org)) sometimes a `Universal Tagset` for English is applied:

| **Tag**      | **Meaning**         | **Examples**                          |
|---------------|---------------------|---------------------------------------|
| ADJ           | adjective           | new, good, high, special, big, local  |
| ADP           | adposition          | on, of, at, with, by, into, under     |
| ADV           | adverb              | really, already, still, early, now    |
| CONJ          | conjunction         | and, or, but, if, while, although     |
| DET           | determiner          | the, a, some, most, every, no         |
| NOUN          | noun                | year, home, costs, time, education    |
| NUM           | number              | twenty\-four, fourth, 1991, 14:24     |
| PRON          | pronoun             | he, their, her, its, my, I, us        |
| PRT           | particle            | at, on, out, over per, that, up, with |
| VERB          | verb                | is, say, told, given, playing, would  |
| .             | punctuation marks   | . , ; !                               |
| X             | other               | ersatz, esprit, dunno, univeristy     |


Some tagsets distinguish quite a lot different tags, some only a few. The resolution depends on 

* the NLP tasks: for some tasks a fine-grained differentiation is not required
* the language: If a language is quite *irregular*, it does not make sense to distinguish PoS in a fine-grained manner, because a tagger would implement all these irregular cases, what may be too complex. For example in German there is no unique rule for the differentiation in *noun-singular* and *noun-plural*. Therefore the Stuttgart-Tübingen-Tagset does not distinguish these two noun-categories. However, in English there is such a rule (append `'s`), which is applicable in nearly all cases. Therefore English Tagsets differentiate these two cases.

## Algorithms for PoS Tagging

PoS-tagging can be implemented in a rule-based or in a data-based approach. As for other NLP methods the rule-based approach is the conventional on. It does not require a training data set, but it requires expert knowledge. Today, data-based approaches are superior, if enough labeled training data is available. Data-based approaches do not require expert knowledge.

In this, first a rule-based approach for tagging is described. Then simple data-based methods, the Unigram and the N-Gram Tagger, are introduced. The currently best performing PoS-taggers learn the tagging-rules from large amounts of PoS-tagged training data by applying machine learning algorithms.

### Rule based Tagging

For rule based-tagging linguistic knowledge on the PoS and patterns that can be applied to determine the PoS of a given word is required. The PoS of a word depends not only on the word itself, e.g. pre- and suffixes, length of the word, etc. but also on surrounding words. Therefore, rules on the word itself, e.g. *does the word end with `ing`*, and rules on the surrounding words, e.g. *is the previous word a determiner (`the`)*, must be defined. An example of a small set of rules is given below. This small set contains rules only on the word itself: 


 ```
 #1. Define Pattern:
 patterns = [
      (r'.*ing$', 'VBG'),               # gerunds
      (r'.*ed$', 'VBD'),                # simple past
      (r'.*es$', 'VBZ'),                # 3rd singular present
      (r'.*ould$', 'MD'),               # modals
      (r'.*\'s$', 'NN$'),               # possessive nouns
      (r'.*s$', 'NNS'),                 # plural nouns
      (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
      (r'the', 'DT'),                   # Determiner
      (r'in','IN'),                     # preposition
      (r'.*', 'NN')                     # nouns (default)
 ]
 
 ```
 
 Once a rule-set (=pattern) is defined, it can be applied by, e.g. a `RegexpTagger`-object from the [NLTK package](https://www.nltk.org) as follows:
 
 ```
 #2.Generate RegexpTagger
 from nltk import RegexpTagger
 regexp_tagger = nltk.RegexpTagger(patterns)
 #3.Tag a sentence. Note that the string, which contains the sentence must be segmented into words
 regexp_tagger.tag("5 friends have been singing in the rain".split())
 ```
The output of this simle Tagger is:
```
[('5', 'CD'),
 ('friends', 'NNS'),
 ('have', 'NN'),
 ('been', 'NN'),
 ('singing', 'VBG'),
 ('in', 'IN'),
 ('the', 'DT'),
 ('rain', 'NN')]
```

### Unigram Tagger

In the context of this lecture a *unigram* is just a single word. A *unigram-tagger* is probably the simplest data-based tagger. As all data-based taggers it requires a labeled training data set (corpus), from which it learns a mapping from a single word to it's PoS:

$$
word \rightarrow PoS(word), \quad \forall word \in V,
$$

where $V$ is the applied vocabulary.

**Training:** For training a Unigram-Tagger a large PoS-tagged corpus is required. Such corpora are publicly available for almost all common languages, e.g. the [Brown Corpus](https://en.wikipedia.org/wiki/Brown_Corpus) for English and the [Tiger Corpus](https://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/tiger/) for German. In such corpora each word is associated with it's PoS, as can be seen in the following sentence from the Brown corpus:

```
[('The', 'DET'), ('Fulton', 'NOUN'), ('County', 'NOUN'), ('Grand', 'ADJ'), ('Jury', 'NOUN'), ('said', 'VERB'), ('Friday', 'NOUN'), ('an', 'DET'), ('investigation', 'NOUN'), ('of', 'ADP'), ("Atlanta's", 'NOUN'), ('recent', 'ADJ'), ('primary', 'NOUN'), ('election', 'NOUN'), ('produced', 'VERB'), ('``', '.'), ('no', 'DET'), ('evidence', 'NOUN'), ("''", '.'), ('that', 'ADP'), ('any', 'DET'), ('irregularities', 'NOUN'), ('took', 'VERB'), ('place', 'NOUN'), ('.', '.')]
```

During training the Unigram-Tagger determines for each word in the corpus which PoS-Tag is associated most often with the word in the training corpus. The result of the training is a table of two columns, the first column is a word and the second the most-frequent PoS of this word:

| Word    | Most Frequent Tag |
|---------|-------------------|
| control | noun              |
| run     | verb              |
| love    | verb              |
| red     | adjective         |
| :       | :                 |



**Tagging:** The learned mapping is the two-column table of word and associated most frequent PoS. This table can be applied to tag each word with it's PoS. 

**Properties:** A unigram tagger is simple to learn and to apply. However, it suffers from the drawback that only the word itself, but not it's context is applied to determine the tag. Consequently a word is always tagged with the same PoS, independent of it's context. Unigram-Tagging is erroneous whenever a PoS applies to a word, which is not the PoS that appeared most often with this word in the training corpora.
 

### N-Gram Tagger

The Unigram-Tagger ignored the context of the word. However, the previous words, or better, the PoS of the previous words, may provide much information for assigning the correct PoS-tag. For example, if the word before `run` is an article, then the PoS-tag of run is probably `noun`, whereas if the word predecessor of `run` is a pronoun, the run's PoS-tag is more likely verb.

A **Bigram-Tagger** assigns the PoS-tag of the current word by taking into account the current word itself and the PoS-tag of the preceiding word. 

$$
(PoS(word_{i-1}),word_i) \rightarrow PoS(word_i), \quad \forall word \in V,
$$

More general, an **N-gram-Tagger** assigns the PoS-tag of the current word by taking into account the current word itself and the PoS-tag of the N-1 preceiding words.   

$$
(PoS(word_{i-N+1}),\ldots,PoS(word_{i-1}),word_i) \rightarrow PoS(word_i), \quad \forall word \in V,
$$

<figure align="center">
<img width="500" src="https://maucher.home.hdm-stuttgart.de/Pics/NGramTagging.png">
<figcaption><b>Figure:</b> A 3-Gram-Tagger determines the PoS-Tag of the current word, by taking into the account the current word and the PoS-Tags of 2 preceiding words.</figcaption>
</figure>

**Training an N-Gram-Tagger:** As for the Unigram-Tagger a large PoS-tagged corpus is required. During training the N-Gram-Tagger determines for each combination of *word plus $N-1$ preceiding PoS-Tags* in the corpus which PoS-Tag is associated most often with the word. The result of the training is a table of $N+1$ columns, the first $N-1$ columns contain the PoS-tags of the preceiding words, followed by a column with the current word and the column, which contains the most frequent PoS-tag for this combination. For example, for a Bigram-Tagger ($N=2$) the table-structure is as follows:

| PoS-Tag of previous word  |  Word   | Most Frequent Tag |
|---------------------------|---------|-------------------|
| article                   | control | noun              |
| pronoun                   | control | verb              |
| pronoun                   | run     | verb              |
| article                   | run     | noun              |
| pronoun                   | love    | verb              |
| article                   | love    | adjective         |
| :                         | :       | :                 |

**Tagging:** The learned mapping is the $N+1$-column table. This table can be applied to tag each *PoS-tag-sequence-word-combination* with the PoS of the current word. 

**Properties:** The larger the $N$, the more context is taken into account and the higher the probability, that the correct PoS-Tag is assigned. However, with an increasing number $N$ also the number of *PoS-tag-sequence-word-combinations$ increases exponentially. Therefore the probability that the text, which must be tagged, contains a combination, which has not been in the training corpus and theirfore is not listed in the mapping table increases. What should be done in the case of such an unknown combination? A standard solution is to train and implement a sequence of N-Gram-Taggers with varying $N$. For example a Unigram-, Bigram-, 3-Gram and 4-Gram-Tagger is trained. For tagging the 4-Gram tagger is applied. If this tagger faces a 4-combination (sequence of 3 PoS-tags plus following word), which is not in it's table, a **Backup-Tagger**, the 3-Gram-Tagger in this case, is applied for this combination. If the corresponding 3-combination is also not in the table of the 3-Gram-Tagger the next Backup-Tagger, which is the Bigram-Tagger, is applied and so on.

In the [next section](02PosTagging.ipynb) the application of all the taggers, described above, is demonstrated. Moreover, in a [previous notebook](../02normalisation/03StemLemma.ipynb) it has already been shown how *TextBlob* can be applied for PoS-Tagging.  
