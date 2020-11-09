TextBlob Stemming and Lemmatization
=====================================

- Author:      Johannes Maucher
- Last update: 2020-09-15

In this notebook the application of stemming and lemmatization shall be demonstrated. For this we apply the correponding modules provided by the NLP package [**TextBlob**](https://textblob.readthedocs.io/en/dev/). Since stemming and lemmatization both require the segmentation of texts into lists of words, segmentation and other preprocessing-functions of TextBlob are also shown. In notebook [Regular Expressions](../01access/05RegularExpressions.ipynb) it has already been demonstrated how to implement segementation in python without additional packages. If you like to go directly to [Word Normalisation click here](#word_normalisation). 

[**TextBlob**](https://textblob.readthedocs.io/en/dev/) is a Python library for Natural Language Processing. It provides a simple API for, e.g.
* Noun phrase extraction
* Part-of-speech tagging
* Sentiment analysis
* Classification
* Language translation and detection powered by Google Translate
* Tokenization (splitting text into words and sentences)
* Word and phrase frequencies
* Parsing
* n-grams
* Word inflection (pluralization and singularization) and lemmatization
* Spelling correction
* WordNet integration

#!pip install textblob

import textblob
print(textblob.__version__)

from textblob import TextBlob

## TextBlob objects
TextBlob objects are like Python strings, which have enhanced with typical NLP processing methods. They are generared as follows:

myBlob1=TextBlob("""TextBlob is a Python (2 and 3) library for processing textual data. 
It provides a simple API for diving into common natural language processing (NLP) tasks 
such as part-of-speech tagging, noun phrase extraction, sentiment analysis, classification, 
translation, and more. New York is a nice city.""")
myBlob2=TextBlob(u"Das Haus steht im Tal auf der anderen Seite des Berges. Es gehört Dr. med. Brinkmann und wird von der Immo GmbH aus verwaltet. Im Haus befindet sich u.a. eine Physio-Praxis.")

## Language Detection

print("Blob1 text language is ",myBlob1.detect_language())
print("Blob2 text language is ",myBlob2.detect_language())

## Sentences and Words
The TextBlob class also integrates methods for tokenisation. The corresponding segmentation of the given text into sentences and words can be obtained as follows:

**English text:**

for s in myBlob1.sentences:
    print("-"*20)
    print(s)

myBlob1.sentences[2].words

As can be seen in the example above multi-word expressions, like *New York* are not detected.

**German text:**

for s in myBlob2.sentences:
    print("-"*20)
    print(s)

As can be seen in the example above, segmentation works fine for English texts, but fails for German texts, in particular if the text contains abbreviations, which are not used in English. However, the [**TextBlob-DE package**](https://pypi.python.org/pypi/textblob-de) provides German language support for *TextBlob*.

## Part-of-Speech Tagging

[Meaning of POS Tags according to Penn Treebank II tagset](https://www.clips.uantwerpen.be/pages/mbsp-tags).

for word,pos in myBlob1.sentences[0].tags:
    print(word, pos)

Doesn't work for German:

for word,pos in myBlob2.sentences[0].tags:
    print(word, pos)

<a id='word_normalisation'></a>
## Word Normalization
In NLP the term *word normalization* comprises methods to map different word forms to a unique form. Word normalization reduces complexity and often improves the NLP task. E.g. in document-classification it usually does not matter in which temporal form a verb is written. The mapping of all temporal forms to the base form reduces the number of words in the vocabulary and likely increases the accuracy of the classifier.

### Singularization
One form of word normalization is to map all words in plural into the corresponding singular form. With *textblob* this can be realized as follows:

myBlob6=TextBlob("The cars in the streets around Main Square have been observed by policemen")
for word in myBlob6.words:
    print(word, word.singularize())

### Lemmatization
Lemmatization maps all distinct forms of a word to the baseform. E.g. the wordforms `went, gone, going` are all mapped to the baseform `go`:

print(TextBlob("went").words[0].lemmatize("v")) # mode "v" for verb
print(TextBlob("gone").words[0].lemmatize("v"))
print(TextBlob("going").words[0].lemmatize("v"))

Lemmatization of adverbs and adjectives:

print(TextBlob("later").words[0].lemmatize("a")) # mode "a" for adverb/adjective
print(TextBlob("prettier").words[0].lemmatize("a"))
print(TextBlob("worse").words[0].lemmatize("a"))

Lemmatizsation of nouns: 

print(TextBlob("women").words[0].lemmatize("n")) # mode "n" for noun

myBlob3=TextBlob("The engineer went into the garden and found the cats lying beneath the trees")

for word in myBlob3.words:
    print(word, word.lemmatize("v"))

### Stemming
The drawback of lemmatization is its complexity. Some languages have a quite regular structure. This means that the different word-forms can be derived from the baseform by a well defined set of rules. In this case the inverse application of these rules can be applied for determining the baseform. However, languages such as German, have many irregular cases. Lemmatization then requires a *dictionary*, which lists all different word-forms an their corresponding baseform.

Stemming is simple and less complex alternative to lemmatization. Stemmers map each word to their stem. In contrast to lemmatization the result of stemming need not be a lexical entry (valid word). Stemmers, e.g. the *Porter Stemmer* apply heuristics for word-suffixes and strip-off found suffixes from the word. E.g. in the word `engineer` a stemmer finds `er` as a frequnet suffix. It strips-off this suffix and outputs the found stem `engin`.  

for word in myBlob3.words:
    print(word, word.stem())

### Word correction
Word correction can also be considered as type of normalization. It maps misspelled words to likely correct form. 

w=TextBlob("sentense is not tru")
wcorr=w.correct()
wcorr

## TextBlob-DE for German Language

#!pip install textblob-de

import textblob_de

myBlob4=textblob_de.TextBlobDE(u"Das Haus steht im Tal auf der anderen Seite des Berges. Es gehört Dr. med. Brinkmann und wird von der Immo GmbH verwaltet. Im Haus befindet sich u.a. eine Physio-Praxis.")

### Sentences and Words

for s in myBlob4.sentences:
    print("-"*20)
    print(s)

myBlob4.sentences[1].words

myBlob4.sentences[1].noun_phrases

### Part-of-Speech Tagging

myBlob5=textblob_de.TextBlobDE("Er ist mit seinen Katzen über drei Tische gesprungen")

for word,tag in myBlob5.tags:
    print(word, tag)

### Lemmatization

for word in myBlob5.words.lemmatize():
    print(word)