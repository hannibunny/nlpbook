# CNN, LSTM and Attention for IMDB Movie Review classification
* Author: Johannes Maucher
* Last Update: 23.11.2020

The IMDB Movie Review corpus is a standard dataset for the evaluation of text-classifiers. It consists of 25000 movies reviews from IMDB, labeled by sentiment (positive/negative). In this notebook a Convolutional Neural Network (CNN) is implemented for sentiment classification of IMDB reviews.

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Embedding, Dense, Input, Flatten, Conv1D, MaxPooling1D, Dropout, Concatenate, GlobalMaxPool1D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import imdb

MAX_SEQUENCE_LENGTH = 500  # all text-sequences are padded to this length
MAX_NB_WORDS = 10000        # number of most-frequent words that are regarded, all others are ignored
EMBEDDING_DIM = 100         # dimension of word-embedding
INDEX_FROM=3

## Access IMDB dataset
The [IMDB dataset](https://keras.io/datasets/) is already available in Keras and can easily be accessed by

`imdb.load_data()`. 

The returned dataset contains the sequence of word indices for each review. 

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=MAX_NB_WORDS,index_from=INDEX_FROM)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

X_train[0][:10] #plot first 10 elements of the sequence

The representation of text as sequence of integers is good for Machine Learning algorithms, but useless for human text understanding. Therefore, we also access the word-index from [Keras IMDB dataset](https://keras.io/api/datasets/imdb/), which maps words to the associated integer-IDs. Since we like to map integer-IDs to words we calculate the inverse wordindex `inv_wordindex`: 

wordindex=imdb.get_word_index(path="imdb_word_index.json")

wordindex = {k:(v+INDEX_FROM) for k,v in wordindex.items()}
wordindex["<PAD>"] = 0
wordindex["<START>"] = 1
wordindex["<UNK>"] = 2
wordindex["<UNUSED>"] = 3

inv_wordindex = {value:key for key,value in wordindex.items()}

The first film-review of the training-partition then reads as follows:

print(' '.join(inv_wordindex[id] for id in X_train[0] ))

Next the distribution of review-lengths (words per review) is calculated:

textlenghtsTrain=[len(t) for t in X_train]

from matplotlib import pyplot as plt

plt.hist(textlenghtsTrain,bins=20)
plt.title("Distribution of text lengths in words")
plt.xlabel("number of words per document")
plt.show()

## Preparing Text Sequences and Labels
All sequences must be padded to unique length of `MAX_SEQUENCE_LENGTH`. This means, that longer sequences are cut and shorter sequences are filled with zeros. For this Keras provides the `pad_sequences()`-function. 

X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)

X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)

Moreover, all class-labels must be represented in one-hot-encoded form:

y_train = to_categorical(np.asarray(y_train))
y_test = to_categorical(np.asarray(y_test))
print('Shape of Training Data Input:', X_train.shape)
print('Shape of Training Data Labels:', y_train.shape)

print('Number of positive and negative reviews in training and validation set ')
print (y_train.sum(axis=0))
print (y_test.sum(axis=0))

## CNN with 2 Convolutional Layers

The first network architecture consists of
* an embedding layer. This layer takes sequences of integers and learns word-embeddings. The sequences of word-embeddings are then passed to the first convolutional layer
* two 1D-convolutional layers with different number of filters and different filter-sizes
* two Max-Pooling layers to reduce the number of neurons, required in the following layers
* a MLP classifier with one hidden layer and the output layer

### Prepare Embedding Matrix and -Layer 

embedding_layer = Embedding(MAX_NB_WORDS,
                            EMBEDDING_DIM,
                            #weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

### Define CNN architecture

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

### Train Network

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['categorical_accuracy'])
model.summary()

print("model fitting - simplified convolutional neural network")
history=model.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=6, verbose=False, batch_size=128)

%matplotlib inline
from matplotlib import pyplot as plt

acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
max_val_acc=np.max(val_acc)

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(8,6))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.grid(True)
plt.legend()
plt.show()

model.evaluate(X_test,y_test)

As shown above, after 6 epochs of training the cross-entropy-loss is 0.475 and the accuracy is 87.11%. However, it seems that the accuracy-value after 3 epochs has been higher, than the accuracy after 6 epochs. This indicates overfitting due to too long learning.

## CNN with different filter sizes in one layer
In [Y. Kim; Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882v2.pdf) a CNN with different filter-sizes in one layer has been proposed. This CNN is implemented below:

![KimCnn](https://maucher.home.hdm-stuttgart.de/Pics/KimCnn.png)

Source: [Y. Kim; Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882v2.pdf)

### Prepare Embedding Matrix and -Layer

embedding_layer = Embedding(MAX_NB_WORDS,
                            EMBEDDING_DIM,
                            #weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

### Define Architecture

convs = []
filter_sizes = [3,4,5]

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

for fsz in filter_sizes:
    l_conv = Conv1D(filters=32,kernel_size=fsz,activation='relu')(embedded_sequences)
    l_pool = MaxPooling1D(4)(l_conv)
    convs.append(l_pool)
    
l_merge = Concatenate(axis=1)(convs)
l_cov1= Conv1D(64, 5, activation='relu')(l_merge)
l_pool1 = GlobalMaxPool1D()(l_cov1)
#l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
#l_pool2 = MaxPooling1D(30)(l_cov2)
l_flat = Flatten()(l_pool1)
l_dense = Dense(64, activation='relu')(l_flat)
preds = Dense(2, activation='softmax')(l_dense)

model = Model(sequence_input, preds)

model.summary()

### Train Network

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['categorical_accuracy'])

print("model fitting - more complex convolutional neural network")
history=model.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=8, batch_size=128)

acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
max_val_acc=np.max(val_acc)

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(8,6))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.grid(True)
plt.legend()
plt.show()

model.evaluate(X_test,y_test)

As shown above, after 8 epochs of training the cross-entropy-loss is 0.467 and the accuracy is 88.47%.

## LSTM

from tensorflow.keras.layers import LSTM, Bidirectional

embedding_layer = Embedding(MAX_NB_WORDS,
                            EMBEDDING_DIM,
                            #weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
l_lstm = Bidirectional(LSTM(64))(embedded_sequences)
preds = Dense(2, activation='softmax')(l_lstm)
model = Model(sequence_input, preds)

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['categorical_accuracy'])

print("model fitting - Bidirectional LSTM")

history=model.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=6, batch_size=128)

acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
max_val_acc=np.max(val_acc)

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(8,6))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.grid(True)
plt.legend()
plt.show()

model.evaluate(X_test,y_test)

As shown above, after 6 epochs of training the cross-entropy-loss is 0.467 and the accuracy is 86.7%. However, it seems that the accuracy-value after 2 epochs has been higher, than the accuracy after 6 epochs. This indicates overfitting due to too long learning.

## Bidirectional LSTM architecture with Attention

### Define Custom Attention Layer
Since Keras does not provide an attention-layer, we have to implement this type on our own. The implementation below corresponds to the attention-concept as introduced in [Bahdanau et al: Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf).

The general concept of writing custom Keras layers is described in the corresponding [Keras documentation](https://keras.io/layers/writing-your-own-keras-layers/). 

Any custom layer class inherits from the layer-class and must implement three methods:

- `build(input_shape)`: this is where you will define your weights. This method must set `self.built = True`, which can be done by calling `super([Layer], self).build()`.
- `call(x)`: this is where the layer's logic lives. Unless you want your layer to support masking, you only have to care about the first argument passed to call: the input tensor.
- `compute_output_shape(input_shape)`: in case your layer modifies the shape of its input, you should specify here the shape transformation logic. This allows Keras to do automatic shape inference.

from tensorflow.keras import regularizers, initializers,constraints
from tensorflow.keras.layers import Layer


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     #name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

embedding_layer = Embedding(MAX_NB_WORDS,
                            EMBEDDING_DIM,
                            #weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
l_gru = Bidirectional(LSTM(64, return_sequences=True))(embedded_sequences)
l_att = Attention(MAX_SEQUENCE_LENGTH)(l_gru)
preds = Dense(2, activation='softmax')(l_att)
model = Model(sequence_input, preds)

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['categorical_accuracy'])

history=model.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=6, batch_size=128)

acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
max_val_acc=np.max(val_acc)

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(8,6))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.grid(True)
plt.legend()
plt.show()

model.evaluate(X_test,y_test)

Again, the achieved accuracy is in the same range as for the other architectures. None of the architectures has been optimized, e.g. through hyperparameter-tuning. However, the goal of this notebook is not the determination of an optimal model, but the demonstration of how modern neural network architectures can be implemented for text-classification.

