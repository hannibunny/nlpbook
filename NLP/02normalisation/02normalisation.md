# Word Normalisation

**Word normalisation** tries to represent words in a unique way. This includes:

* Character case: In most of the NLP tasks it should not be distinguished whether a word or characters of a word are written in upper or lower case.[^F1]
* Word correction: Spelling mistakes shall be corrected, such that only correct words constitute the vocabulary
* In case of spelling ambiguities, all words shall be mapped to a unique spelling (i.e. valid spellings due to German Rechtschreibreform)  
* In many NLP tasks different forms shall not be distinguished. For example in Information Retrieval (Web Search) the temporal form of a verb should be ignored as well as singular or plural of nouns. 

```{admonition} Example
:class: dropdown
Consider a search-engine like [google](https://www.google.com/). If you enter your query-words, you like to get the same results independent of the wordform. E.g. the query `video encoding` shall provide the same result as `encode videos`. For this all words of the query and all words in the index of the search-engine must be normalised to a unique form. This normalisation does not only provide better results, but it also reduces memory- and time-complexity, because the index is much smaller than without normalisation  
```

## Morphology
Morphology is the study of **words**, how they are formed, and their relationship to other words in the same language[^F2].


### Inflection and Word-Formation
**Lexeme and word-form:** A lexeme is a set of inflected word-forms. For instance, the lexeme eat contains the word-forms *eat*, *eats*, *eaten*, and *ate*. *Eat* and *eats* are thus considered different word-forms belonging to the same lexeme eat. Eat and Eater, on the other hand, are different lexemes, as they refer to two different concepts. Usuaully, two kinds of morphological rules are distinguished. Some morphological rules relate to **different forms of the same lexeme**, while other rules relate to **different lexemes**. Rules of the first kind are **inflectional rules**, while those of the second kind are rules of **word formation**.[^F2].

<figure align="center">
<img width="600" src="https://maucher.home.hdm-stuttgart.de/Pics/MorphologieZwergBeispiel.png">
<figcaption><b>Figure:</b> Inflection (Flexion) and Word-formation in German.</figcaption>
</figure>

Word-formation can further be distinguished into derivation and compounding (Komposition). The subcategories of inflexion are 

* Deklination: Nouns, adjectives, articles, pronouns are declined by gender, number and case. (house, houses, mouse, mice, he, she, it, ...)
* Conjugation: Verbs are conjugated by appendig affixes, which may express tense, mood, voice, aspect, person, or number (I go, he goes, we went, I am going, ...)
* Comparative and superlative of adjectives (small, smaller, the smallest)


### Stemming and Lemmatisation

A **morphological parser**, determines for an arbitrary word-form the morphemes and possibly some morphological properties. For example for the given word *cities* a morphological parser may return the morphem *city* and the properties *Noun* and *Plural*.

Frequently used subcategories of morphological parsers are **Stemming** and **Lemmatisation**. Both are described in the following subsection. 

Stemming and lemmatization are applied for *text normalisation*. Their goal is to reduce inflectional forms and sometimes derivationally related forms of a word to a common base form. E.g. 


**Lemmatisation** finds the baseform (also called the *lemma*) of a word, e.g.:

* go, went, gone, going, goes $\Rightarrow$ go
* mouse, mice, mouse's $\Rightarrow$ mouse
* smaller, smallest $\Rightarrow$ small
* worse $\Rightarrow$ small
* factories $\Rightarrow$ factory

The result of lemmatization is always a valid word. In order to find the baseform a lemmatizer applies morphological rules and dictionaries (for the irregular cases).

**Stemming** finds for given words a corresponding *stem*. E.g.[^F3]

* education $\Rightarrow$ educ
* expensive $\Rightarrow$ expens
* exploration $\Rightarrow$ explor
* cities $\Rightarrow$ citi
* better $\Rightarrow$ better

Stemmers are often implemented in a quite crude way. For example they apply heuristics in order to determine and cut off suffixes and prefixes of words. The determined stem is usually not a valid word. The benefit of stemmers compared to lemmatizers is their simplicity and hence the reduced time to calculate the new representation. Popular Stemmers are [Porter Stemmer](https://de.wikipedia.org/wiki/Porter-Stemmer-Algorithmus), [Snowball Stemmer](https://snowballstem.org) and [Lancaster Stemmer](https://www.scientificpsychic.com/paice/paice.html). 

```{note}
Tasks like information retrieval (web-search) or document classification may benefit from normalized word representations calculated by stemming and lemmatisation. However, it may also decrease accuracy. In the end, there is no way around evaluating both options in the respective application.   
```


 

[^F1]: An exception is e.g. the task of Part-Of-Speech Tagging in German. Here, the case may provide helpful information.

[^F2]: [wikipedia.org](https://en.wikipedia.org/wiki/Morphology_(linguistics))

[^F3]: Examples have been calucalted by the [NLTK porter stemmer online version](http://textanalysisonline.com/nltk-porter-stemmer)