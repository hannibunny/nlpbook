Correction of Spelling Errors
========================

* Author: Johannes Maucher
* Last Update: 2020-09-09

## Overall Process

Correction of misspelled words requires
* the detection of an error
* the determination of likely candidates for the correct word
* the determination of the nearest correct word, given the erroneous word

The detection of an erroneous word can be done by just checking if the given word is contained in the underlying vocabulary. If the word is not contained in the vocabulary a **statistical language model** (see chapter xxx) is applied in order to determine the most probable words at the given position for the given context - the context is typically given by the neighbouring words. For the set of most probable valid words at the position of the erroneous words the nearest one is determined. For this a method to determine `distance` (or `similarity`) is required. In the following section the most common method to measure distance between two character-sequences, the Levensthein-Distance, is described.

## Levensthein Distance

The `Minimum Edit Distance` (MED) between two character-sequences $\underline{a}$ and $\underline{b}$ is defined to be the minimum number of edit-operations to transform  $\underline{a}$ into $\underline{b}$. The set of edit operations is *Insert (I), Delete (D) and Replace (R)* a single character. Since *R* can be considered to be a combination of D and I, it may make sense to define the costs of *R* to be 2, whereas the costs of *I* and *D* are 1 each. In this case (where *R* has double costs), the distance-measure is called `Levensthein Distance`.  

```{admonition} Example
:class: dropdown
The Levensthein Distance between *aply* and *apply* is 1, because one *Insert (I)* is required to transform the first into the second character-sequence. The Levensthein Distance between *aply* and *apple* is 3, because one *Insert (I)* and one *Replace (R)* is required.  
```

### Algorithm for Calculating Levensthein Distance:

The calculation of the Levensthein Distance is a little bit more complex than calculating other distance measures such as the `euclidean distance` or the `cosine similarity`. The fastest algorithm for calculating the Levensthein-Distance between a source-sequence $\underline{a}$ and a target-sequence $\underline{b}$ belongs to the category of [Dynamic Programming approach](https://en.wikipedia.org/wiki/Dynamic_programming). 

The Dynamic Programming algorithm for calculating the Levensthein distance starts with the initialisation, which is depicted in the figure below for the example, that the distance between the source word INTENTION and the target word EXECUTION shall be calculated.

<figure align="center">
<img width="700" src="https://maucher.home.hdm-stuttgart.de/Pics/medTableEmpty.png">
<figcaption> <b>Figure:</b> Initialisation of Algorithm for computing Levensthein Distance between source (INTENTION) and target (EXECUTION).</figcaption>
</figure>



**Initialisation:**

* The characters of the source word plus a preceding # (which indicates the empty sequence) define the rows of the distance-matrix $D$ in down-to-top order and the characters of the target word plus a preceding # define the columns of this matrix (left-to-right).
* The first column $D[i,0]$ of the matrix is initialised by

$$
D[i,0]=i \quad \forall i \in [0,n] 
$$

and the first row $D[0,j]$ is initialised by 

$$
D[0,j]=j \quad \forall j \in [0,m], 
$$

where $n$ is the length of the source- and $m$ is the length of the target-word.


**Rekursion:**

The goal is to fill the matrix $D$ recursively, such that entry $D[i,j]$ in row $i$, column $j$ is the Levensthein distance between the first $i$ characters of the source- and the first $j$ characters of the target word. For this we fill the matrix from the lower-left to the upper right-corner as follows: For the next empty field three options are considered and finally the best of them, i.e. the one with minimal costs, defines the new entry in the matrix.

1.) This empty field can be passed horizontally by moving from the left neighbour to the current positions. This move corresponds to an Insert-operation, which costs 1. Hence, the new cost (distance) in this field would be the cost of the left neighbour+1.

2.) The empty field can also be passed vertically by moving from the lower neighbour to the current positions. This move corresponds to a Deletion-operation, which costs 1. Hence, the new cost (distance) in this field would be the cost of the lower neighbour+1.

3.) The empty field can also be passed diagonally by moving from the lower-left neighbour to the current positions. This move corresponds either to a Replace-operation (if source-character in row i and target-character in column j are different) or to NO-Edit-Operation (if source-character in row i and target-character in column j are the same). In the first case the new costs would be costs of the lower-left neighbour + 2, in the second case the new costs are the same as in the lower-left neighbour.

From these 3 potential costs, the minimum is selected and inserted in entry $D[i,j]$. Formally, the recursive algorithm is as follows:

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
As far as the entire distance matrix is filled by the recursion described above, the Levensthein-distance is determined to be the entry at the upper-right corner:

$$D_{lev}(\underline{a},\underline{b})=D[n,m]$$

The filled distance matrix for the given example is depicted below:

<figure align="center">
<img width="670" src="https://maucher.home.hdm-stuttgart.de/Pics/medTableFull.png">
<figcaption><b> Figure:</b> At the end the Levensthein distance is the number at the upper right corner in the matrix. In this example the distance between INTENTION and EXECUTION has been calculated to be 8.</figcaption>
</figure>

### Implementation  of Levensthein Algorithm

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

