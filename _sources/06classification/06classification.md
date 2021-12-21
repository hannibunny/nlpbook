# Text Classification

Many NLP tasks apply classification of texts. A given sequence of words, e.g. a sentence, utterance, question, section or document must be assigned to a pre-defined classes, e.g. sentiment, question-type, intend, document-type etc. A prominent category of document-classification is Spam-Filtering. Here the input-text is usually an Email and the classes to which this input must be assigned are spam and non-spam.

As many other NLP tasks, text-classification can be realized by implementing a rule-based or a data-based approach (see {doc}`../intro` ). Since today data-based approaches nearly always outperform rule-based approaches, we just consider data-based approaches here. Data-based document-classification is realized by **Supervised Machine Learning**. 

The general concept of Supervised Machine Learning is depicted in the figure below. 

<figure align="center">
<img width="600" src="https://maucher.home.hdm-stuttgart.de/Pics/SupervisedLearningSchemaEnglish.png">
<figcaption>Training and Inference in Supervised Machine Learning</figcaption>
</figure>

**Training:** Supervised training requires many pairs of *(input,label)*. For each input the corresponding label must be known. In the case of text classification the input is a text and the label indicates to which class this text belongs to. For example in spam-filtering the input is the text of an email and the label is either *spam* or *no-spam*. From the set of pairs *(input,label)* a general mapping from input to label is learned by the machine learning algorithm. This mapping is also called the learned model. This model can than be applied in  the Inference face in order to determine for each input the corresponding class-label. As can be seen in the figure above, the raw-input must be transformed by a *Feature Extraction* into a numeric vector. In the case of document classification this numerich vector may be the BoW-Vector of the document.

**Inference:** Once the model has been learned from the training-data, it can be applied for classification. Arbitrary raw-inputs are transformed into their numeric vector representations. The vector is passed to the model, which determines the most likely class-index.

In the **conventional approach** for text-classification the numeric vector, which is passed to the Machine-Learning algorithm is the BoW-vector and the Machine Learning algorithm itself may be any supervised learning algorithm, e.g. Logistic Regression, Decision Tree, Naive Bayes, Support-Vector-Machine, Random Forest, conventional neural networks etc. This conventional approach may be sufficient for relatively simple classification tasks, in particular tasks, for which the order of words in the text is not relevant, e.g. topic-classification (decide whether a given document belongs to category *tech*, *general news*, *poetry*, ...). However, for more complex tasks, in particular tasks for which word-order is relevant, such as sentiment analysis, the conventional approach is outperformed by approaches, which

* apply sequences of **word-embeddings** instead of BoW-vectors at the input of the ML-algorithm
* **(Deep) Neural Networks** such as Convolutional Neural Networks (CNNs) or Recurrent Neural Networks (RNN) of different flavors, such as Simple RNN, LSTM or GRU.

<figure align="center">
<img width="800" src="https://maucher.home.hdm-stuttgart.de/Pics/overAllPicture.png">
<figcaption>From the conventional approach (BoW and conventional ML) to Word Embeddings and Deep Neural Networks</figcaption>
</figure>

In this chapter one frequently used conventional approach - Naive Bayes Classification - is introduced. Deep Neural Network solutions are presented in later chapters.  

