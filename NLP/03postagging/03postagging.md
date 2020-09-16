Part-Of-Speech Tagging
======================

A *Part-of-Speech (PoS)* is a category of words, that have similar grammatical properties. A simplified set of Part-of-Speeches contains `nouns`, `verbs`, `adjectives`, `adverbs`, `determiners`, etc. For some tasks more detailed categories such as `nouns-singular`, `nouns-plural`, `proper noun`, etc. are applied. 

In NLP the process of assigning PoS to the given words in a sentence is called **PoS tagging**. Depending of the word's definition and it's context (the surrounding words) a **PoS tagger** tries to assign the correct PoS to each word in the given text. PoS-tagging is challenging because many words can have different PoSs, e.g. run, love, ... 

Knowing the PoS of the words is of benefit for many applications, e.g.

* once the PoSs of all words in a sentence are known, a **syntax parser** can calculate the syntactic tree of the sentence and it can determine if the syntactic structure is correct (syntax correction in editors)
* from the PoS and the position within the syntax-tree the role/function of the word can be and derived, which is a key input form semantic analysis
* in some NLP applications only words belonging to certain PoSs are required. E.g. for sentiment-analysis adjectives and adverbs are informative, but not determiners.  

<figure align="center">
<img width="300" src="https://maucher.home.hdm-stuttgart.de/Pics/parseTreeMorningFlight.jpg">
<figcaption><b>Figure:</b> Syntax tree: In order to calculate this syntactic structure, the PoS of each word must be known in advance.</figcaption>
</figure>

In this section common PoS and Pos-Tagsets are introduced as well as algorithms for PoS-Tagging. Moreover, it is demonstrated how packages like *TextBlob* or *spaCy* can be applied for PoS-Tagging. 

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
