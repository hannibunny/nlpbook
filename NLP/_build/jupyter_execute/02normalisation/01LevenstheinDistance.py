# Correction of Spelling Errors

* Author: Johannes Maucher
* Last Update: 2020-09-09

## Overall Process

Correction of misspelled words requires
* the detection of an error
* the determination of likely candidates for the correct word
* the determination of the nearest correct word, given the erroneous word

The detection of an erroneous word can be done by just checking if the given word is contained in the underlying vocabulary. If the word is not contained in the vocabulary a **statistical language model** (see chapter xxx) is applied in order to determine the most probable words at the given position for the given context - the context is typically given by the neighbouring words. For the set of most probable valid words at the position of the erroneous words the nearest one is determined. For this a method to determine `distance` (or `similarity`) is required. In the following section the most common method to measure distance between two character-sequences, the Levensthein-Distance, is described.

## Levensthein Distance

The `Minimum Edit Distance` (MED) between two character-sequences $\overrightarrow{s_1}$ and $\overrightarrow{s_2}$ is defined to be the minimum number of edit-operations to transform  $\overrightarrow{s_1}$ into $\overrightarrow{s_2}$. The set of edit operations is *Insert (I), Delete (D) and Replace (R)* a single character. Since *R* can be considered to be a combination of D and R, it may make sense to define the costs of *R* to be 2, whereas the costs of *I* and *D* are 1 each. In this case (where *R* has double costs), the distance-measure is called `Levensthein Distance`.  

```{admonition} Example
:class: dropdown
The Levensthein Distance between *aply* and *apply* is 1, because one *Insert (I)* is required to transform the first into the second character-sequence. The Levensthein Distance between *aply* and *apple* is 3, because one *Insert (I)* and one *Replace (R)* is required.  
```

### Define function for Levensthein distance:

**Initialising:**
$$
D[i,0]=i \quad \forall i \in [0,n] 
$$
$$
D[0,j]=j \quad \forall j \in [0,m] 
$$

**Rekursion**

For all $i \in [1,n]$:
       For all $j \in [1,m]$:
$$
D[i,j]=\min \left\{
\begin{array}{l}
D[i-1,j]+1 \\
D[i,j-1]+1 \\
D[i-1,j-1] + \left\{ 
\begin{array}{lll} 
2 & if & \underline{a}_i \neq \underline{b}_j \\    
0 & if & \underline{a}_i = \underline{b}_j 
\end{array}
\right.
\end{array}
\right.
$$
**Termination**

$$MED(\underline{a},\underline{b})=D[n,m]$$

<figure align="center">
<img width="700" src="https://maucher.home.hdm-stuttgart.de/Pics/medTableEmpty.png">
<figcaption>NLP Processing Chain</figcaption>
</figure>


<figure align="center">
<img width="670" src="https://maucher.home.hdm-stuttgart.de/Pics/medTableFull.png">
<figcaption>NLP Processing Chain</figcaption>
</figure>

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

Apply the function defined above for calculation of Levensthein distance between two words:

print(levdist("rakete","rokete"))



