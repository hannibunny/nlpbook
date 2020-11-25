# Text Classification Application: Fake News detection
* Author: Johannes Maucher
* Last update: 24.11.2020

In this notebook conventional Machine Learning algorithms are applied to learn a discriminator-model for distinguishing fake- and non-fake news.

What you will learn:
* Access text from .csv file
* Preprocess text for classification
* Calculate BoW matrix
* Apply conventional machine learning algorithms for fake news detection
* Evaluation of classifiers

## Access Data
In this notebook a [fake-news corpus from Kaggle](https://www.kaggle.com/c/fake-news/data) is applied for training and testing Machine Learning algorithms. Download the 3 files and save it in a directory. The path of this directory shall be assigned to the variable `path`in the following code-cell: 

import pandas as pd
pfad="/Users/johannes/DataSets/fake-news/"
train = pd.read_csv(pfad+'train.csv',index_col=0)
test = pd.read_csv(pfad+'test.csv',index_col=0)
test_labels=pd.read_csv(pfad+'submit.csv',index_col=0)

Data in dataframe `train` is applied for training. The dataframe `test`contains the texts for testing the model and the dataframe `test_labels` contains the true labels of the test-texts. 

print("Number of texts in train-dataframe: \t",train.shape[0])
print("Number of columns in train-dataframe: \t",train.shape[1])
train.head()

Append the test-dataframe with the labels, which are contained in dataframe `test_labels`.

test["label"]=test_labels["label"]

print("Number of texts in test-dataframe: \t",test.shape[0])
print("Number of columns in test-dataframe: \t",test.shape[1])
test.head()

## Data Selection

In the following code cells, first the number of missing-data fields is determined. Then the information in columns `author`, `title` and `text` are concatenated to a single string, which is saved in the column `total`. After this process, only columns `total` and `label` are required, all other columns can be removed in the `train`- and the `test`-dataframe. 

train.isnull().sum(axis=0)

test.isnull().sum(axis=0)

train = train.fillna(' ')
train['total'] = train['title'] + ' ' + train['author'] + ' ' + train['text']

train = train[['total', 'label']]

train.head()

test = test.fillna(' ')
test['total'] = test['title'] + ' ' + test['author'] + ' ' + test['text']
test = test[['total', 'label']]

## Preprocessing
The input texts in column `total` shall be preprocessed as follows:
* stopwords shall be removed
* all characters, which are neither alpha-numeric nor whitespaces, shall be removed
* all characters shall be represented in lower-case.
* for all words, the lemma (base-form) shall be applied

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

stop_words = stopwords.words('english')

lemmatizer = WordNetLemmatizer()
for index in train.index:
    #filter_sentence = ''
    sentence = train.loc[index,'total']
    # Cleaning the sentence with regex
    sentence = re.sub(r'[^\w\s]', '', sentence)
    # Tokenization
    words = nltk.word_tokenize(sentence)
    # Stopwords removal
    words = [lemmatizer.lemmatize(w).lower() for w in words if not w in stop_words]
    filter_sentence = " ".join(words)
    train.loc[index, 'total'] = filter_sentence

First 5 cleaned texts in the training-dataframe:

train.head()

Clean data in the test-dataframe in the same way as done for the training-dataframe above:

lemmatizer = WordNetLemmatizer()
for index in test.index:
    #filter_sentence = ''
    sentence = test.loc[index,'total']
    # Cleaning the sentence with regex
    sentence = re.sub(r'[^\w\s]', '', sentence)
    # Tokenization
    words = nltk.word_tokenize(sentence)
    # Stopwords removal
    words = [lemmatizer.lemmatize(w).lower() for w in words if not w in stop_words]
    filter_sentence = " ".join(words)
    test.loc[index, 'total'] = filter_sentence

First 5 cleaned texts in the test-dataframe:

test.head()

## Determine Bag-of-Word Matrix for Training- and Test-Data
In the code-cells below two different types of Bag-of-Word matrices are calculated. The first type contains the **term-frequencies**, i.e. the entry in row $i$, column $j$ is the frequency of word $j$ in document $i$. In the second type, the matrix-entries are not the term-frequencies, but the tf-idf-values. 

Note that for a given typ (term-frequency or tf-idf) a separate matrix must be calculated for training and testing. Since we always pretend, that only training-data is known in advance, the matrix-structure, i.e. the columns (= words) depends only on the training-data. This matrix structure is calculated in the row:

```
count_vectorizer.fit(X_train)
```
and
```
tfidf.fit(freq_term_matrix_train),
```
respectively. An important parameter of the `CountVectorizer`-class is `min_df`. The value, which is assigned to this parameter is the minimum frequency of a word, such that it is regarded in the BoW-matrix. Words, which appear less often are disregarded.

The training data is then mapped to this structure by 
```
count_vectorizer.transform(X_train)
```
and
```
tfidf.transform(X_train),
```
respectively.

For the test-data, however, no new matrix-structure is calculated. Instead the test-data is transformed to the structure of the matrix, defined by the training data.

X_train = train['total'].values
y_train = train['label'].values

X_test = test['total'].values
y_test = test['label'].values

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

Train BoW-models and transform training-data to BoW-matrix:

count_vectorizer = CountVectorizer(min_df=4)
count_vectorizer.fit(X_train)
freq_term_matrix_train = count_vectorizer.transform(X_train)
tfidf = TfidfTransformer(norm = "l2")
tfidf.fit(freq_term_matrix_train)
tf_idf_matrix_train = tfidf.transform(freq_term_matrix_train)

freq_term_matrix_train.toarray().shape

tf_idf_matrix_train.toarray().shape

Transform test-data to BoW-matrix:

freq_term_matrix_test = count_vectorizer.transform(X_test)
tf_idf_matrix_test = tfidf.transform(freq_term_matrix_test)

## Train a linear classifier
Below a [Logistic Regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression) is trained. This is just a linear classifier with a sigmoid- or softmax- activity-function. 

X_train=tf_idf_matrix_train
X_test=tf_idf_matrix_test
#X_train=freq_term_matrix_train
#X_test=freq_term_matrix_test

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

## Evaluate trained model
First, the trained model is applied to predict the class of the training-samples:

y_pred_train = logreg.predict(X_train)

y_pred_train

from sklearn.metrics import classification_report

The model's prediction are compared with the true classes of the training-samples. The classification-report contains the common metrics for evaluating classifiers:

print(classification_report(y_train,y_pred_train))

The output of the classification report shows, that the model is well fitted to the training data, since it predicts training data with an accuracy of 98%.

However, accuracy on the training-data, provides no information on the model's capability to classify new data. Therefore, below the model's prediction on the test-dataset is calculated:

y_pred_test = logreg.predict(X_test)

print(classification_report(y_test,y_pred_test))

The model's accuracy on the test-data is weak. The model is overfitted on the training-data. It seems that the distribution of test-data is significantly different from the distribution of training-data. 

The main drawback in this experiment is possibly the application of the BoW-model to represent texts. BoW disregards word-order and semantic relations between words. The application of word-embeddings and neural networks like CNNs and LSTMs may perform much better.

train.head()

from tensorflow.keras.preprocessing import text

MAX_NB_WORDS=5000

tokenizer=text.Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(train["total"])

trainSeq=tokenizer.texts_to_sequences(train["total"])

testSeq=tokenizer.texts_to_sequences(test["total"])

tokenizer.num_words

textlenghtsTrain=[len(t) for t in trainSeq]

textlenghtsTest=[len(t) for t in testSeq]

from matplotlib import pyplot as plt

plt.hist(textlenghtsTrain,bins=20)
plt.title("Distribution of text lengths in words")
plt.xlabel("number of words per document")
plt.show()

textlenghtsTrain.sort(reverse=True)

textlenghtsTrain[:10]

MAX_SEQUENCE_LENGTH=800
EMBEDDING_DIM=100

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

X_train = pad_sequences(trainSeq, maxlen=MAX_SEQUENCE_LENGTH)
X_test = pad_sequences(testSeq, maxlen=MAX_SEQUENCE_LENGTH)

y_train = to_categorical(np.asarray(train["label"]))
y_test = to_categorical(np.asarray(test["label"]))

from tensorflow.keras.layers import Embedding, Dense, Input, Flatten, Conv1D, MaxPooling1D, Dropout, Concatenate, GlobalMaxPool1D
from tensorflow.keras.models import Model

embedding_layer = Embedding(MAX_NB_WORDS,
                            EMBEDDING_DIM,
                            #weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
l_cov1= Conv1D(32, 5, activation='relu')(embedded_sequences)
l_pool1 = MaxPooling1D(2)(l_cov1)
l_cov2 = Conv1D(64, 3, activation='relu')(l_pool1)
l_pool2 = MaxPooling1D(5)(l_cov2)
l_flat = Flatten()(l_pool2)
l_dense = Dense(64, activation='relu')(l_flat)
preds = Dense(2, activation='softmax')(l_dense)
model = Model(sequence_input, preds)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['categorical_accuracy'])
model.summary()

history=model.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=6, verbose=True, batch_size=128)

