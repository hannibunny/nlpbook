Word Normalisation
==================

**Word normalisation** tries to represent words in a unique way. This includes:

* Character case: In most of the NLP tasks it should not be distinguished whether a word or characters of a word are written in upper or lower case.[^F1]
* Word correction: Spelling mistakes shall be corrected, such that only correct words constitute the vocabulary
* In case of spelling ambiguities, all words shall be mapped to a unique spelling (i.e. valid spellings due to German Rechtschreibreform)  
* In many NLP tasks different forms shall not be distinguished. For example in Information Retrieval (Web Search) the temporal form of a verb should be ignored as well as singular or plural of nouns. 

```{admonition} Example
:class: dropdown
Consider a search-engine like [google](https://www.google.com/). If you enter your query-words, you like to get the same results independent of the wordform. E.g. the query `video encoding` shall provide the same result as `encode videos`. For this all words of the query and all words in the index of the search-engine must be normalised to a unique form. This normalisation does not only provide better results, but it also reduces memory- and time-complexity, because the index is much smaller than without normalisation  
```



[^F1]: An exception is e.g. the task of Part-Of-Speech Tagging in German. Here, the case may provide helpful information.