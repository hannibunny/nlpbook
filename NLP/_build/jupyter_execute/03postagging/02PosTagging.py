# POS Tagging with NLTK

* Author: Johannes Maucher
* Last update: 18.09.2020

Required modules:

from nltk.corpus import brown
from nltk import FreqDist
from nltk import word_tokenize
from nltk import RegexpTagger
from nltk import tag
import nltk

## Regular Expression POS Tagging
Define regular-expression rules, that will be applied for POS tagging.

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

Generate a *RegexpTagger*-object with the defined rules. 

regexp_tagger = RegexpTagger(patterns)

Apply the RegexpTagger for tagging a single sentence:

regexp_tagger.tag("5 friends have been singing in the rain".split())

Apply the *RegexpTagger* for tagging the first 3 sentences of the brown corpus.

regexp_tagger.tag(brown.sents()[3])

Evaluate the tagger using category _news_ of the brown corpus. The `evaluate()`-method returns the accuracy (i.e. the rate of correct Tag-assignments) of the tagger on this test-corpus.

brown_tagged_sents=brown.tagged_sents(categories='news')
print(regexp_tagger.evaluate(brown_tagged_sents))

## Unigram Tagger

from nltk import UnigramTagger, DefaultTagger, BigramTagger
from nltk import FreqDist,ConditionalFreqDist

A UnigramTagger requires a tagged corpus. From the tagged corpus it learns a mapping from word to pos-tag by determining for each word the most frequent tag in the corpus. The trained tagger then assigns to each word the most frequent pos-tag as determined in the training corpus.

In this notebook the pos-tagged Brown Corpus is applied. The tagset used in this corpus is quite sophisticated. It can be obtained by the following command:

nltk.help.brown_tagset()

However, NLTK provides a method to map the most common tagsets to a simple [universal POS Tagset](http://www.nltk.org/book/ch05.html). 

| Tag | Meaning | English Examples | 
| --- | --- | --- | 
| ADJ | adjective | new, good, high, special, big, local | 
| ADP | adposition | on, of, at, with, by, into, under | 
| ADV | adverb | really, already, still, early, now | 
| CONJ | conjunction | and, or, but, if, while, although | 
| DET | determiner, article | the, a, some, most, every, no, which | 
| NOUN | noun | year, home, costs, time, Africa | 
| NUM | numeral | twenty-four, fourth, 1991, 14:24 | 
| PRT | particle | at, on, out, over per, that, up, with | 
| PRON | pronoun | he, their, her, its, my, I, us | 
| VERB | verb | is, say, told, given, playing, would | 
| . | punctuation marks | . , ; ! | 
| X | other | ersatz, esprit, dunno, gr8, univeristy |

The Brown Corpus with the simple universal tagset can be obtained as follows:

brown_tagged_sents=brown.tagged_sents(tagset="universal")
print(brown_tagged_sents[:1])

A *UnigramTagger*-object is generated and trained with the Brown Corpus with universal tagset:

complete_tagger=UnigramTagger(train=brown_tagged_sents)

The trained Unigram-Tagger is applied to tag a single sentence:

mySent1="the cat is on the mat".split()
print(complete_tagger.tag(mySent1))

Compare tags assigned by the Unigram-Tagger and the tags assigned by the current NLTK standard tagger on a single sentence:

mySent2="This is major tom calling ground control from space".split()
print("Unigram Tagger: \n",complete_tagger.tag(mySent2))
print("\nCurrent Tagger applied for NLTK pos_tag(): \n",nltk.pos_tag(mySent2,tagset='universal'))

The performance of the trained tagger is evaluated on the same corpus as applied for training. The performance measure is the rate of words that have been tagged correctly.

print("Performance of complete Tagger: ",complete_tagger.evaluate(brown_tagged_sents))

The rate of correctly taggged words is quite high. However, this method of evaluation is not valid, since the same corpus has been used for evaluation as for training. Therefore we split the corpus into a *training-part* and a *test-part*. The *UnigramTagger* is then trained with the *training-part* and evaluated with the disjoint  *test-part*.

size = int(len(brown_tagged_sents) * 0.9)
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]
unigram_tagger = UnigramTagger(train_sents,backoff=DefaultTagger("NN"))
print("Performance of Tagger with 90% Training and 10% Testdata: ",unigram_tagger.evaluate(test_sents))

As expected the rate of correctly tagged words is lower, but this value is now a valid evaluation measure.

### Unigram Tagger, which applies only frequent words for training
A trained Unigram Tagger must store a table, which assigns to each word the most frequent POS-tag. Since this table can be quite large, an option is to train the Unigram Tagger only with the most frequent words. 
The following code generates a list of different Unigram taggers, each with an other amount of frequent words of the brown corpus. The plot visualizes the Unigram Tagger performance in dependence of the number of most frequent words, stored in the tagger. Note that in the code below the _UnigrammTagger_ is initialized with a dictionary of tagged words, whereas in the code above the _UnigrammTagger_ is initialized with a corpus of tagged words. Both options are possible. 

def display():
    import pylab
    words_by_freq = FreqDist(brown.words(categories='news')).most_common(2**15)
    cfd = ConditionalFreqDist(brown.tagged_words(categories='news'))
    sizes = 2 ** pylab.arange(15)
    perfs = [performance(cfd, words_by_freq[:size]) for size in sizes]
    pylab.plot(sizes, perfs, '-bo')
    pylab.title('Lookup Tagger Performance with Varying Model Size')
    pylab.xlabel('Model Size')
    pylab.ylabel('Performance')
    pylab.show()

def performance(cfd, wordlist):
    lt = dict((word[0], cfd[word[0]].max()) for word in wordlist)
    baseline_tagger = UnigramTagger(model=lt, backoff=DefaultTagger('NN'))
    return baseline_tagger.evaluate(brown.tagged_sents(categories='news'))


display()

## N-Gram Tagger
Unigram taggers assign to each wort $w_n$ the tag $t_n$, which is the most frequent tag for $w_n$ in the training corpus. N-Gram taggers are a generalization of Unigram-Taggers. During training they determine for each combination of $N-1$ previous tags $t_{n-1},t_{n-2},...$ and the current word $w_n$ the most frequent tag $t_n$. Tagging is then realized, by inspecting the $n-1$ previous tags and the current word $w_n$ and assigning the most frequent tag, which appeared for this combination in the training corpus.  
![NgramTagging](https://maucher.home.hdm-stuttgart.de/Pics/NGramTagging.png)

baseline=nltk.DefaultTagger('NOUN')
unigram = UnigramTagger(train=train_sents,backoff=baseline)
bigram = BigramTagger(train=train_sents,backoff=unigram)

bigram.evaluate(test_sents)

# Find most frequent nouns
The most frequent nouns usually provide information on the subject of a text. Below, the most frequent nouns of an already tagged text of the *Treebank*-corpus are determined. Let's see if we can conclude the text's subject.  

from nltk.corpus import treebank
from nltk import FreqDist
from nltk import bigrams

print("\nTreebank sentences: ", treebank.sents(fileids="wsj_0003.mrg"))

tagged0003=treebank.tagged_words(tagset="universal",fileids="wsj_0003.mrg")
print("File tagged0003: ",tagged0003)

fdist=FreqDist(a[0].lower() for a in tagged0003 if a[1]=="NOUN")
#fdist.tabulate(20)
print(fdist.most_common(20))
freqNouns = [w[0] for w in fdist.most_common(20)]
fdist.plot(20)

Next, the adjectives immediately before the most frequent nouns are determined. What can be concluded from them? 

taggedPairs=bigrams(tagged0003)
adjNounPairs=[(a[0],b[0]) for (a,b) in taggedPairs if b[0].lower() in freqNouns and a[1]=="ADJ"]
for a in adjNounPairs:
    print(a)