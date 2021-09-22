#!/usr/bin/env python
# coding: utf-8

# # Text Classification Application: Fake News detection
# * Author: Johannes Maucher
# * Last update: 24.11.2020
# 
# In this notebook conventional Machine Learning algorithms are applied to learn a discriminator-model for distinguishing fake- and non-fake news.
# 
# What you will learn:
# * Access text from .csv file
# * Preprocess text for classification
# * Calculate BoW matrix
# * Apply conventional machine learning algorithms for fake news detection
# * Evaluation of classifiers

# ## Access Data
# In this notebook a [fake-news corpus from Kaggle](https://www.kaggle.com/c/fake-news/data) is applied for training and testing Machine Learning algorithms. Download the 3 files and save it in a directory. The path of this directory shall be assigned to the variable `path`in the following code-cell: 

# In[1]:


import pandas as pd
pfad="/Users/johannes/DataSets/fake-news/"
train = pd.read_csv(pfad+'train.csv',index_col=0)
test = pd.read_csv(pfad+'test.csv',index_col=0)
test_labels=pd.read_csv(pfad+'submit.csv',index_col=0)


# Data in dataframe `train` is applied for training. The dataframe `test`contains the texts for testing the model and the dataframe `test_labels` contains the true labels of the test-texts. 

# In[2]:


print("Number of texts in train-dataframe: \t",train.shape[0])
print("Number of columns in train-dataframe: \t",train.shape[1])
train.head()


# Append the test-dataframe with the labels, which are contained in dataframe `test_labels`.

# In[3]:


test["label"]=test_labels["label"]


# In[4]:


print("Number of texts in test-dataframe: \t",test.shape[0])
print("Number of columns in test-dataframe: \t",test.shape[1])
test.head()


# ## Data Selection
# 
# In the following code cells, first the number of missing-data fields is determined. Then the information in columns `author`, `title` and `text` are concatenated to a single string, which is saved in the column `total`. After this process, only columns `total` and `label` are required, all other columns can be removed in the `train`- and the `test`-dataframe. 

# In[5]:


train.isnull().sum(axis=0)


# In[6]:


test.isnull().sum(axis=0)


# In[7]:


train = train.fillna(' ')
train['total'] = train['title'] + ' ' + train['author'] + ' ' + train['text']


# In[8]:


train = train[['total', 'label']]


# In[9]:


train.head()


# In[10]:


test = test.fillna(' ')
test['total'] = test['title'] + ' ' + test['author'] + ' ' + test['text']
test = test[['total', 'label']]


# ## Preprocessing
# The input texts in column `total` shall be preprocessed as follows:
# * stopwords shall be removed
# * all characters, which are neither alpha-numeric nor whitespaces, shall be removed
# * all characters shall be represented in lower-case.
# * for all words, the lemma (base-form) shall be applied

# In[11]:


import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re


# In[12]:


stop_words = stopwords.words('english')


# In[13]:


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


# First 5 cleaned texts in the training-dataframe:

# In[14]:


train.head()


# Clean data in the test-dataframe in the same way as done for the training-dataframe above:

# In[15]:


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


# First 5 cleaned texts in the test-dataframe:

# In[16]:


test.head()


# ## Determine Bag-of-Word Matrix for Training- and Test-Data
# In the code-cells below two different types of Bag-of-Word matrices are calculated. The first type contains the **term-frequencies**, i.e. the entry in row $i$, column $j$ is the frequency of word $j$ in document $i$. In the second type, the matrix-entries are not the term-frequencies, but the tf-idf-values. 
# 
# Note that for a given typ (term-frequency or tf-idf) a separate matrix must be calculated for training and testing. Since we always pretend, that only training-data is known in advance, the matrix-structure, i.e. the columns (= words) depends only on the training-data. This matrix structure is calculated in the row:
# 
# ```
# count_vectorizer.fit(X_train)
# ```
# and
# ```
# tfidf.fit(freq_term_matrix_train),
# ```
# respectively. An important parameter of the `CountVectorizer`-class is `min_df`. The value, which is assigned to this parameter is the minimum frequency of a word, such that it is regarded in the BoW-matrix. Words, which appear less often are disregarded.
# 
# The training data is then mapped to this structure by 
# ```
# count_vectorizer.transform(X_train)
# ```
# and
# ```
# tfidf.transform(X_train),
# ```
# respectively.
# 
# For the test-data, however, no new matrix-structure is calculated. Instead the test-data is transformed to the structure of the matrix, defined by the training data.

# In[17]:


X_train = train['total'].values
y_train = train['label'].values


# In[18]:


X_test = test['total'].values
y_test = test['label'].values


# In[19]:


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


# Train BoW-models and transform training-data to BoW-matrix:

# In[20]:


count_vectorizer = CountVectorizer(min_df=4)
count_vectorizer.fit(X_train)
freq_term_matrix_train = count_vectorizer.transform(X_train)
tfidf = TfidfTransformer(norm = "l2")
tfidf.fit(freq_term_matrix_train)
tf_idf_matrix_train = tfidf.transform(freq_term_matrix_train)


# In[21]:


freq_term_matrix_train.toarray().shape


# In[22]:


tf_idf_matrix_train.toarray().shape


# Transform test-data to BoW-matrix:

# In[23]:


freq_term_matrix_test = count_vectorizer.transform(X_test)
tf_idf_matrix_test = tfidf.transform(freq_term_matrix_test)


# ## Train a linear classificator
# Below a [Logistic Regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression) is trained. This is just a linear classifier with a sigmoid- or softmax- activity-function. 

# In[24]:


X_train=tf_idf_matrix_train
X_test=tf_idf_matrix_test
#X_train=freq_term_matrix_train
#X_test=freq_term_matrix_test


# In[25]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# ## Evaluate trained model
# First, the trained model is applied to predict the class of the training-samples:

# In[26]:


y_pred_train = logreg.predict(X_train)


# In[27]:


y_pred_train


# In[28]:


from sklearn.metrics import confusion_matrix, classification_report


# In[29]:


confusion_matrix(y_train,y_pred_train)


# The model's prediction are compared with the true classes of the training-samples. The classification-report contains the common metrics for evaluating classifiers:

# In[30]:


print(classification_report(y_train,y_pred_train))


# The output of the classification report shows, that the model is well fitted to the training data, since it predicts training data with an accuracy of 98%.
# 
# However, accuracy on the training-data, provides no information on the model's capability to classify new data. Therefore, below the model's prediction on the test-dataset is calculated:

# In[31]:


y_pred_test = logreg.predict(X_test)


# In[32]:


confusion_matrix(y_test,y_pred_test)


# In[33]:


print(classification_report(y_test,y_pred_test))


# The model's accuracy on the test-data is weak. The model is overfitted on the training-data. It seems that the distribution of test-data is significantly different from the distribution of training-data. This hypothesis can be verified by ignoring the data from `test.csv` and instead split data from `train.csv` into a train- and a test-partition. In this modified experiment performance on test-data is much better, because the texts within `train.csv` origin from the same distributions. 
