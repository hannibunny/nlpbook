#!/usr/bin/env python
# coding: utf-8

# # Validation of Classifiers
# 
# <img src="http://maucher.home.hdm-stuttgart.de/Pics/SupervisedLearningSchemaValidation.png" style="width:700px" align="center">

# Text classification is part of supervised Machine Learning (ML). As depicted in the picture above, supervised ML relies on labeled data - not only for training, but also for model testing. It is important that training- and test-datasets are disjoint.
# 
# Once a model is trained, it is applied on test-data and calculates predictions for the test-data input. Since for test-data also the true label (output) is known, these true labels can be compared with the predicted labels. Based on this comparison, different  **metrics** for classifier evaluations can be calculated. The most important classifier metrics are described below.

# Assume that for 10 test-data samples the true and predicted labels (class-indeces) are as listed in the table below:
# 
# | True Label | Predicted Label |
# |:----------:|:---------------:|
# |      0     |        0        |
# |      0     |        1        |
# |      1     |        1        |
# |      1     |        1        |
# |      1     |        0        |
# |      0     |        0        |
# |      1     |        0        |
# |      0     |        0        |
# |      0     |        0        |
# |      0     |        0        |
# 
# All of the metrics described below, can be calculated from this comparison. 
# 
# ## Confusion matrix
# 
# The confusion matrix contains for each pair of classes $i$ and $j$, the number of class $i$ elements, which have been predicted to be class $j$. Usually, each row corresponds to a true-class label and each column corresponds to a predicted class label. 
# 
# In the general 2-class confusion matrix, depicted below, the class labels are $P$ (positive) and $N$. The matrix entries are then
# 
# * **TP (True Positives):** Number of samples, which belong to class $P$
#  and have correctly been predicted to be class $P$
#  
# * **TN (True Negative):** Number of samples, which belong to class $N$
#  and have correctly been predicted to be class $N$
#  
# * **FP (False Positives):** Number of samples, which belong to class $N$
#  but have falsely been predicted to be class $P$
#  
# * **FN (False Negatives):** Number of samples, which belong to class $P$
#  but have falsely been predicted to be class $N$
#  
# <img src="http://maucher.home.hdm-stuttgart.de/Pics/confusionMatrixTN.png" style="width:200px" align="center">
# 
# For the given example of predicted and true class labels the confusion matrix is:
# 
# <img src="http://maucher.home.hdm-stuttgart.de/Pics/confusionMatrixExample.png" style="width:200px" align="center">
# 
# 
# ## Accuracy
# 
# Accuracy is the ratio of correct predictions among all predictions. For the 2-class problem and the labels $P$ and $N$, accuracy can be calculated from the entries of the confusion matrix:
# 
# $$
# Acc =\frac{TP+TN}{TP+TN+FP+FN}
# $$
# 
# 
# In the example the accuracy is
# $$
# Acc =\frac{5+2}{5+2+1+2}=0.7
# $$
# 
# 
# ## Recall
# The recall of class $i$ is the ratio of correctly predicted class $i$ elements, among all elements, which truly belong to class $i$. The recall of class $P$ is:
# 
# $$
# Rec_P =\frac{TP}{TP+FN}
# $$
# 
# and for class $N$:
# 
# $$
# Rec_N =\frac{TN}{TN+FP}
# $$
# 
# In the example:
# 
# $$
# Rec_1=\frac{2}{4} \quad \mbox{ and } \quad Rec_0=\frac{5}{6}
# $$
# 
# 
# ## Precision 
# 
# The precision of class $i$ is the ratio of true class $i$ elements, among all elements, which have been predicted to be class $i$. The precision of class $P$ is:
# 
# 
# $$
# Pre_P =\frac{TP}{TP+FP}
# $$
# 
# and for class $N$:
# 
# $$
# Pre_N =\frac{TN}{TN+FN}
# $$
# 
# In the example:
# 
# $$
# Pre_1=\frac{2}{3}  \quad \mbox{ and } \quad Pre_0=\frac{5}{7}
# $$
# 
# ## F1-Score
# 
# The F1-score of a class $i$ is the harmonic mean of this class' precision and recall:
# 
# $$
# F1_i = 2 \cdot \frac{Pre_i \cdot Rec_i}{Pre_i + Rec_i}
# $$
# 
# In the example: 
# 
# $$
# F1_1= 2 \cdot \frac{\frac{2}{4} \frac{2}{3}}{\frac{2}{4} + \frac{2}{3}}  \quad \mbox{ and } \quad F1_0= 2 \cdot \frac{\frac{5}{6} \frac{5}{7}}{\frac{5}{6} + \frac{5}{7}} 
# $$
# 

# In[ ]:




