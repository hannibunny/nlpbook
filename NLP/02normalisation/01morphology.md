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

 

[^F2]: [wikipedia.org](https://en.wikipedia.org/wiki/Morphology_(linguistics))

[^F3]: Examples have been calucalted by the [NLTK porter stemmer online version](http://textanalysisonline.com/nltk-porter-stemmer)