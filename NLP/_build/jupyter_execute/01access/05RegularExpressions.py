# Regular expressions in Python  

- Author:      Johannes Maucher
- Last update: 2020-09-09

This notebook demonstrates the application of regular expressions in Python.  

## Regular Expressions Operators and Characterclass Symbols
|Operator|Behavior|
|--- |--- |
|.|Wildcard, matches any character|
|^abc|Matches some pattern abc at the start of a string|
|abc$|Matches some pattern abc at the end of a string|
|[abc]|Matches one of a set of characters|
|[A-Z0-9]|Matches one of a range of characters|
|*|Zero or more of previous item, e.g. a*, [a-z]* (also known as Kleene Closure)|
|+|One or more of previous item, e.g. a+, [a-z]+|
|?|Zero or one of the previous item (i.e. optional), e.g. a?, [a-z]?|
|{n}|Exactly n repeats where n is a non-negative integer|
|{n,}|At least n repeats|
|{,n}|No more than n repeats|
|{m,n}|At least m and no more than n repeats|
|a([aA])+|Parentheses that indicate the scope of the operators|

In addition to the operators listed above the `|`- operator is also frequently used. It acts as a disjunction, for example
`|ed|ing|s|` matches to all character sequences with either `ed`, `ing` or `s`. 

For frequently applied character classes, the following shortcut-symbols are defined:

|Symbol|Function|
|--- |--- |
|\d|Any decimal digit (equivalent to [0-9])|
|\D|Any non-digit character (equivalent to [^0-9])|
|\s|Any whitespace character (equivalent to [ \t\n\r\f\v])|
|\S|Any non-whitespace character (equivalent to [^ \t\n\r\f\v])|
|\w|Any alphanumeric character (equivalent to [a-zA-Z0-9_])|
|\W|Any non-alphanumeric character (equivalent to [^a-zA-Z0-9_])|
|\b|Boundary between word and non-word|


## Definition of patterns
Regular expressions are patterns of character sequences. In Python such patterns must be defined as **raw strings**. A raw string is a character sequence, which is not interpreted. A string is defined as raw-string, by the prefix `r`. For example

`mypattern = r"[0-9]+\s"`

is a raw string pattern, which matches to all sequences which consist of one ore more decimal numbers, followed by a white-space character.

## Matching and Searching
The two main application categories of regular expressions are matching and searching: 

* In matching applications the pattern represents a syntactic rule and an arbitrary character sequence (text) is parsed, if it is consistent to this syntax. For example in an web-interface, where users can enter their date of birth a regular expression can be applied to check if the user entered the date in an acceptable format.
* In search applications the pattern defines a type of character-sequence, which is searched for in an arbitrary long text. 

The most important methods of the Python regular expression package `re` are:

* `re.findall(pattern,text)`: Searches in the string-variable `text`- for all matches with `pattern`. All found non-overlapping matches are returned as a list of strings.
* `re.split(pattern,text)`: Searches in the string-variable `text`- for all matches with `pattern`. At all found patterns the text is split. A list of all splits is returned.
* `re.search(pattern,text)`: Searches in the string-variable `text`- for all matches with `pattern`. The first found match is returned as a *match-object*. The return value is `None`, if no matches are found.
* `re.match(pattern,text)`: Checks if the first characters of the string-variable `text` match to the pattern. If this is the case a *match-object* is returned, otherwise `None`.
* `re.sub(pattern,replacement,text)`: Searches for all matches of pattern in text and replaces this matches by the string `replacement`.


import nltk
import re

dummytext1="His email-address is foo.bar@bar-foo.com but he doesn't check his emails frequently"

Assume, that we have texts, that contain email-addresses, like the dummy-text above. The task is to find all email addresses in the text. For this we can define a pattern, which matches to syntactically correct email-addresses and pass this pattern to the `findall(pattern,text)`-method of the `re`-package:

mailpattern=r'[\w_\.-]+@[\w_\.-]+\.\w+'
re.findall(mailpattern,dummytext1)

dummytext2="This is just a dummy test, which is applied for demonstrating regular expressions in Python. The current date is 2017-10-17."

Find first word, which begins with character *d*:

pattern1=r"\s[d]\S*\s"
search_result=re.search(pattern1,dummytext2)
if search_result:
    print(search_result.group())
else:
    print("Pattern not in Text")

Find all words, which begin with character *d*:

search_result=re.findall(pattern1,dummytext2)
if search_result:
    print(search_result)
else:
    print("Pattern not in Text")

Find all words, which begin with character *d*. Return only the words, not the whitespaces around them:

pattern2=r"\s([d]\S*)\s"
search_result=re.findall(pattern2,dummytext2)
if search_result:
    print(search_result)
else:
    print("Pattern not in Text")

Same result as above:

pattern3=r"\b[d]\S*\b"
search_result=re.findall(pattern3,dummytext2)
if search_result:
    print(search_result)
else:
    print("Pattern not in Text")

Replace substrings:

templateText="Dear Mrs. <Name>, we like to invit you to our <Event>"
name="Keane"
event="summer school"
re.sub(r"<Event>",event,re.sub(r"<Name>",name,templateText))


## Segmentation into words using regular expressions

import nltk
import re

text="""Es wäre schön, wenn wir mal wieder fußballspielen könnten.
Oder was meint ihr? Ich bin jederzeit bereit. Thomas habe ich bereits gefragt.
Er meinte: "Ich schau mir lieber das Spiel der Bayern gegen den 1.FC Köln
in der Allianz-Arena an." """

#cleanedTokens=re.findall(r"[\wäüöÄÜÖß]+",text)  # Einfache Lösung die funktioniert
cleanedTokens = re.split(r"[\s.,;:()!?\"]+", text)
#cleanedTokens = text.split()
#cleanedTokens = re.findall(r"[^\s.,;:()!?\"]+", text) #liefert gleiches Ergebnis wie Split, jedoch ohne den leeren String
#cleanedTokens = re.findall(r"\w+[.]\w+|[^\s.,;:()!?\"]+", text) #Damit wird auch Punkt innerhalb eines Wortes erlaubt

ts=set(cleanedTokens)
print("Anzahl der unterschiedlichen Tokens: ",len(ts))
print("------------Die Menge der Worte--------------")
for t in sorted(ts):
    print(t.strip("\".,?:"))

## Segmentation into words using NLTK

print("-"*20+"Test von nltk.regexp_tokenize")
text=text
ntokens=nltk.regexp_tokenize(text,r"[\wäöüÄÖÜß]+")
for a in ntokens:
    print(a)
print("Anzahl der Tokens:     ",len(ntokens))

print("-"*20+"Test von nltk.word_tokenize")
ntokens=nltk.word_tokenize(text)
for a in ntokens:
    print(a)
print("Anzahl der Tokens:     ",len(ntokens))

## Segmentation into sentences using regular expressions

text1="""Es wäre schön, wenn wir mal wieder fußballspielen könnten.
Oder was meint ihr? Ich bin jederzeit bereit. Thomas habe ich bereits gefragt.
Er meinte: "Ich schau mir lieber das Spiel der Bayern gegen den 1.FC Köln 
in der Allianz-Arena an." """

Sents=re.split(r"[?!.:]+\s",text1)  # Einfache Lösung die funktioniert
for t in Sents:
    print('-'*10)
    print(t)
    #print 
print('-'*10)
print("Anzahl der Tokens:                   ",len(Sents))

## Segmentation into sentences using NLTK

text2="""Dr. med. R. Steiner meint es wäre schön, wenn wir mal wieder fußballspielen könnten.
Oder was meint ihr? Ich bin jederzeit bereit. Thomas usw. habe ich bereits gefragt.
Er meinte: "Ich schau mir lieber das Spiel der Bayern gegen den 1.FC Köln 
in der Allianz-Arena an.\" Es schauen 7 Mio. zu, inkl. mir selbst. U.a. soll auch Messi etc. dabei sein.\""""

sent_tokenizer=nltk.data.load('tokenizers/punkt/german.pickle')
sents = sent_tokenizer.tokenize(text2)
for a in sents:
    print('-'*10)
    print(a)
print('-'*10)
print("Anzahl der Tokens:                   ",len(sents))