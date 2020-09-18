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
 #2.Generate RegexpTagger and tag a sentence
 from nltk import RegexpTagger
 regexp_tagger = nltk.RegexpTagger(patterns)
 # Tag a sentence. Note that the string, which contains the sentence must be segmented into words
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

### N-Gram Tagger
