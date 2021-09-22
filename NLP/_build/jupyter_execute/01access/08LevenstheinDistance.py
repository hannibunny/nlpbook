#!/usr/bin/env python
# coding: utf-8

# # Levensthein Distance Calculation
# 
# * Author: Johannes Maucher
# * Last Update: 2020-09-09
# 
# Define function for Levensthein distance:

# In[1]:


def levdist(first, second):
    if len(first) > len(second):
        first, second = second, first
    if len(second) == 0:
        return len(first)
    first_length = len(first) + 1
    second_length = len(second) + 1
    distance_matrix = [[0] * second_length for x in range(first_length)]
    for i in range(first_length):
       distance_matrix[i][0] = i
    for j in range(second_length):
       distance_matrix[0][j]=j
    for i in range(1, first_length):
        for j in range(1, second_length):
            deletion = distance_matrix[i-1][j] + 1
            insertion = distance_matrix[i][j-1] + 1
            substitution = distance_matrix[i-1][j-1]
            if first[i-1] != second[j-1]:
                substitution += 2 #Use +=1 for Minimum Edit Distance
            distance_matrix[i][j] = min(insertion, deletion, substitution)
    return distance_matrix[first_length-1][second_length-1]


# Apply the function defined above for calculation of Levensthein distance between two words:

# In[2]:


print(levdist("rakete","rokete"))


# In[ ]:





# In[ ]:




