# Introduction


* Author: Prof. Dr. Johannes Maucher
* Institution: Stuttgart Media University
* Document Version: 0.1
* Last Update: 09.09.2020

## What is NLP?

Natural Language Processing (NLP) strieves to build computers, such that they can understand and generate natural language. Since computers usually only understand formal languages (programming languages, assembler, etc), NLP techniques must provide the transformation from natural language to a formal language and vice versa.

<figure align="center">
<img width="300" src="https://maucher.home.hdm-stuttgart.de/Pics/NLUnderstandingGeneration.jpg">
<figcaption>Transformation between natural and formal language</figcaption>
</figure>

This lecture focuses on the direction from natural language to formal language. However, in the later chapters also techniques for automatic language generation are explained. In any case, only natural language in written form is considered. Speech recognition, i.e. the process of transforming speech audio signals into written text, is not in the scope of this lecture.   

As a science NLP is a subfield of Artificial Intelligence, which itself belongs to Computer Science. In the past linguistic knowledge has been a key-komponent for NLP. 

<figure align="center">
<img width="300" src="https://maucher.home.hdm-stuttgart.de/Pics/NLPundAndereWissenschaften.jpg">
<figcaption>Sciences, used by NLP</figcaption>
</figure>

The old approach of NLP, the so called **Rule-based-approach** can be described by representing linguistic rules in a formal language and **parsing** text according to this rule. In this way, e.g. the syntactic structure of sentences can be derived and from the syntactic structure semantics are infered.

The enormous success of NLP during the last few years is based on **Data-based-approaches**, which increasingly substitute the old Rule-based-approach. The idea of this approach is to learn language statistics from large amounts of digitally available texts (**copora**). For this, modern **Machine Learning (ML)** techniques, such as Deep Neural Networks are applied. The learned statistics can then be applied e.g. for *Part-of-Speech-Tagging*, *Named-Entity-Recognition*, *Text Summarisation*, *Semantic Analysis*, *Language Translation*, *Text Generation*, *Question-Answering*, *Dialog-Systems* and many other NLP tasks.

As the picture below describes, Rule-based-approaches require expert-knowledge of the linguists, whereas Data-based approaches require large amount of data, ML-algorithms and performant Hardware.

<figure align="center">
<img width="500" src="https://maucher.home.hdm-stuttgart.de/Pics/RegelVsDatenAnsatz.png">
<figcaption>Rule-based and data-based approach</figcaption>
</figure>

The following statement of Fred Jelinek expresses the increasing dominance of Data-based-approaches:
 

```{epigraph}
Every time I fire a linguist, the performance of the speech recognizer goes up.

-- Fred Jelinek[^F1]
```

[^F1]: Frederick Jelinek (18 November 1932 – 14 September 2010) was a Czech-American researcher in information theory, automatic speech recognition, and natural language processing.

```{admonition} Example
:class: dropdown
Consider the NLP task Spam Classification. In a Rule-based approach one would define rules like *if text contains Viagra then class=spam*, *if sender address is part of a given black-list then class=spam*, etc. In a Data-based-approach such rules are not required. Instead a large corpus of e-mails labeled with either *spam* or *ham* is required. A Machine Learning Algorithm, like e.g. a Naive Bayes Classifier, will learn a statistical model from the given training data. The learned model can then be applied for spam-classification.  
```


## NLP Process Chain and Lecture Contents

In order to realize NLP tasks one usually has to implement a chain of processing steps for accessing, storing and preprocessing before the task specific model can be learned and applied. The following figure sketches an entire processing chain in general.

<figure align="center">
<img width="800" src="https://maucher.home.hdm-stuttgart.de/Pics/nlpProcessChain.png">
<figcaption>NLP Processing Chain</figcaption>
</figure>

This processing chain defines the **content of this lecture**:

1. Methods for **accessing text** from different types of sources
2. Text **preprocessing** like segmentation, normalisation, POS-tagging, etc
3. Models for **representing words and texts**.
4. Statistical **Language Models**.
4. Architectures for implementing **NLP tasks** such as word-completion, auto-correction, information retrieval, document classification, automatic translation, automatic text generation, etc.   

The lecture has a practical focus, i.e. for most of the techniques the implementation in Python is demonstrated.

## The challenge of ambiguity

In contrast to many formal languages (programming languages), natural language is ambiguous on different levels:
* Segmentation: Shall *Stamford Bridge* be segmentated into 2 words? Is *Web 2.0* one expression?...
* Homonyms (ball) and Synonyms (bike and bycicle)
* Part-of-Speech: Is *love* a verb, an adjective or a noun?
* Syntax: The sentence *John saw the man on the mountain with a telescope* has multiple syntax trees.
* Semantic: *We saw her dug* (*Wir sahen ihre Ente* oder *Wir sahen, wie sie sich geduckt hat*)
* Ambiguity of pronouns, e.g. *The bottle fell into the glass. It broke.*

## Some popular NLP applications

* Spam Filter / Document Classification
* Sentiment Analysis / Trend Analysis
* Automatic Correction of Words and Syntax (e.g. in Word)
* Auto completion (WhatsApp, Search Engine)
* Information Retrieval / Web Search
* Automatic Text Generation: [Sportberichterstellung](https://www.retresco.de), [Open AI's GPT](https://www.spektrum.de/news/kuenstliche-intelligenz-der-textgenerator-gpt-3-als-sprachtalent/1756796).
* Text Summarisation
* Automatic Translation
* Question Answering, Dialogue Systems, Digital Assitants, Chatbots
* Personal Profiling, e.g. for employment [ZEIT-Artikel zum automatischen Recruiting](https://www.zeit.de/2018/35/kuenstliche-intelligenz-vorstellungsgespraech-interview-test)
* Political orientation, e.g. for election campaigns [Cambridge Analytica and Michael Kosinski](https://www.tagesanzeiger.ch/ausland/europa/diese-firma-weiss-was-sie-denken/story/17474918)
* Recommender Systems 

 
