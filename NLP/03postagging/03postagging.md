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


