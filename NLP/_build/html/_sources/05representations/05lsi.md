### Latent Semantic Indexing (LSI)

LSI has been developed in {cite}`Deerwester1990` as a method for topic-extraction. For the description of LSI, assume that we have 5 simple documents, which contain the follwing words:

* document d1: *cosmonaut, moon, car*
* document d2: *astronaut, moon*
* document d3: *cosmonaut*
* document d4: *car, truck*
* document d5: *car*
* document d6: *truck*

If we construct the BoW-matrix for these documents and transpose this matrix, the result is called *Inverted-Index*. In the example the Inverted Index is: 

$$
\left(
\begin{array}{l|cccccc}
& d1 & d2 & d3 & d4 & d5 & d6 \\
\hline
cosmonaut & 1 & 0 & 1 & 0 & 0 & 0 \\
astronaut & 0 & 1 & 0 & 0 & 0 & 0 \\
moon & 1 & 1 & 0 & 0 & 0 & 0 \\
car & 1 & 0 & 0 & 1 & 1 & 0 \\
truck & 0 & 0 & 0 & 1 & 0 & 1 \\
\end{array}
\right)
$$

Another representations of the form $\mathbf{w}=\mathbf{A}\cdot \mathbf{d}$ is given below. The matrix $\mathbf{A}$ is called **term-by-document matrix**.

$$
\left(
\begin{array}{c}
cosmonaut \\
astronaut \\
moon \\
car \\
truck \\
\end{array}
\right)
=
\left(
\begin{array}{cccccc}
 1 & 0 & 1 & 0 & 0 & 0 \\
 0 & 1 & 0 & 0 & 0 & 0 \\
 1 & 1 & 0 & 0 & 0 & 0 \\
 1 & 0 & 0 & 1 & 1 & 0 \\
 0 & 0 & 0 & 1 & 0 & 1 \\
\end{array}
\right)
\cdot
\left(
\begin{array}{c}
d1 \\
d2 \\
d3 \\
d4 \\
d5 \\
d6 \\
\end{array}
\right)
$$

In the Inverted-Index each column represents a document. Each column is an $n$-dimensional vector, where $n$ is the number of words in the corpus. Hence, each document can be considered as a point in the n-dimensional space. LSI defines a transformation from this $n$-dimensional space into a $k$-dimensional space, where $k$ is usually much smaller than $n$. The representation of the documents in this new $k$-dimensional space is such that, documents which refer to the same topics, have similar k-dimensional vectors (are nearby points in the $k$-dimensional space). 

For example the 2-dimensional space, the 6 documents of the example may have the following representations:[^f1]

$$
\left(
\begin{array}{l|cccccc}
& d1 & d2 & d3 & d4 & d5 & d6 \\
\hline
\{cosmonaut,astronaut,moon\} & 1 & 1 & 1 & 0 & 0 & 0 \\
\{car,truck\} & 1 & 0 & 0 & 1 & 1 & 1 \\
\end{array}
\right).
$$

[^f1]: Note that this representation is not the true result of the transformation, but a simplified one, just for desribing the idea of the approach.

As can be seen in this 2-dimensional vector space document 2 and 3 are described by the same vector $(1,0)$, even though in the original 6-dimensional space their vectors are orthogonal to each other, because these documents have no word in common.

As sketched in the matrix above, the new dimensions in the 2-dimensional space, do not belong to single words, but to topics and each topic is represented by a list of words, which frequently appear in the documents, which belong to this topic. More accurate: The new dimensions (topics) in the $k$-dimensional space are linear combinations of the old dimensions (words) in the $n$-dimensional space.

Note that this implies a different understanding of *semantically related words*, than the concept implied by Distributional Semantic Models (DSM). In DSMs two words are semantically related, if they frequently appear in the same context, where context is given by the surrounding words. In LSI two words are "semantically related", if they frequently appear in the same documents.

Latent Semantic Indexing (LSI) applies **Singular Value Decomposition (SVD)** for calculating the low-dimensional topic space {cite}`Manning2000`. SVD calculates a factorisation of the term-by-document matrix $\mathbf{A}$:

$$
A_{t \times d} = T_{t \times n} S_{n \times n} (D_{d \times n})^T,
$$ (svd-factors)

where $t$ and $d$ are the number of words and the number of documents, respectively and $n=min(t,d)$. The factor matrices have the following properties:

* columns in $T$ are orthonormal
* columns in $D$ are orthonormal
* in $S$ only elements on the main diagonal are non-zero.

The elements on the main diagonal of $S$ are the so called *Singular Values* in decreasing order. The Singular Values reflect the variance of data along the corresponding dimension. The 3 factor matrices define a rotation of the original space, such that 

* the first dimension of the new space is defined by the direction, along which data varies maximal,
* the second dimension of the new space is defined by the direction, which is orthogonal to the first, and belongs to the second strongest variance, 
* the third dimension of the new space is defined by the direction, which is orthogonal to the first two dimensions, and belongs to the third strongest variance, 
* ...

```{note}
SVD can be considered to be a generalisation of Principal Component Analysis (PCA) in the sense that SVD can also be applied to non-square matrices. PCA calculates the Eigenvectors and Eigenvalues of the covariance-matrix of the given data. The Eigenvectors with the strongest associated Eigenvalues are the Principal Components, i.e. the directions, along which data-variance is maximal. Similar as Eigenvalues in PCA, the Singular Values of SVD belong to the directions of maximal data variance.
```

The SVD matrix factorisation applied to the term-by-document matrix of the example yields the following 3 factors:

$$
T=
\left(
\begin{array}{l|ccccc}
& dim1 & dim2 & dim3 & dim4 & dim5  \\
\hline
cosmonaut&  0.44 & -0.3 & -0.57 & 0.58 & -0.25 \\
astronuat&   0.13 & -0.33 & 0.59 & -0.0 & -0.73 \\
moon&   0.48 & -0.51  & 0.37 &-0.0 &   0.61 \\
car&   0.7 &  0.35 & -0.15 &-0.58 &-0.16 \\
truck&   0.26 & 0.65 & 0.41 & 0.58 & 0.09 \\
 \end{array}
\right)
$$ (svd-T)

$$
S=
\left(
\begin{array}{ccccc}
 2.16 & 0 & 0 & 0 & 0 \\
 0 & 1.59 & 0 & 0 & 0 \\
 0 & 0 & 1.28 & 0 & 0 \\
 0 & 0 & 0 & 1.00 & 0 \\
 0 & 0 & 0 & 0 & 0.39 \\
 \end{array}
\right)
$$ (svd-S)

$$
D^T=
\left(
\begin{array}{l|cccccc}
& d1 & d2 & d3 & d4 & d5 & d6 \\
\hline
dim1 & 0.75& 0.28& 0.2&  0.45& 0.33& 0.12\\
dim2 & -0.29&-0.53&-0.19& 0.63& 0.22& 0.41\\
dim3 & -0.28& 0.75&-0.45& 0.2& -0.12& 0.33\\
dim4 & -0.0&   0.0&   0.58& 0.0 & -0.58& 0.58\\
dim5 &  0.53 &-0.29&-0.63& -0.19&-0.41& 0.22\\ 
  \end{array}
\right)
$$ (svd-D)


Given these factors, calculated by SVD, how do we obtain a lower-dimensional space from the original $n$-dimensional space, where $n$ is the number of words in the corpus?

As indicated by the subscripts of the matrix-factors in {eq}`svd-factors`

* matrix $T$ has $n$ columns
* matrix $S$ has $n$ rows and $n$ columns
* matrix $D$ has $n$ columns  

The lower-dimensional space is now obtained by

1. Select the number of dimensions $k$ of the lower-dimensional space. Note that $t$ is the number of topics, that shall be distinguished.
2. Keep the first $k$ columns of matrix $T$ and remove the other $n-k$ columns to obtain the reduced matrix $T'$. Silmilarly, keep the first $k$ rows and the first $k$ columns of matrix $S$ to obtain the reduced matrix $S'$ and keep the first $k$ columns of matrix $D$ and to obtain the reduced matrix $D'$.


	$$
	T'=
	\left(
	\begin{array}{l|cc}
	& dim1 & dim2  \\
	\hline
	cosmonaut&  0.44 & -0.3 \\
	astronaut&   0.13 & -0.33 \\
	moon&   0.48 & -0.51 \\
	car&   0.7 &  0.35 \\
	truck&   0.26 & 0.65  \\
	\end{array}
	\right)
	$$ (svd-Tneu)
	
	$$
	S'=
	\left(
	\begin{array}{cc}
	 2.16 & 0  \\
	 0 & 1.59 \\
	 \end{array}
	\right)
	$$ (svd-Sneu)

	$$
	D'^T=
	\left(
	\begin{array}{l|cccccc}
	& d1 & d2 & d3 & d4 & d5 & d6 \\
	\hline
	dim1 & 0.75& 0.28& 0.2&  0.45& 0.33& 0.12\\
	dim2 & -0.29&-0.53&-0.19& 0.63& 0.22& 0.41\\
	\end{array}
	\right)
	$$ (svd-Dneu)

3. Then 

	$$
	A'=T' \cdot S' \cdot D'^T
	$$

	is the best approximation, in terms of *least square error*, of the original term-by-document matrix $A$. 
	
4. the columns of the **Matrix $B=S' \cdot D'^T$** are the coordinates of the  documents in the $k-$dimensional **latent semantic space**

$$
B=
\left(
\begin{array}{l|cccccc}
& d1 & d2 & d3 & d4 & d5 & d6 \\
\hline
dim1 & 1.62 & 0.60 & 0.44&  0.97& 0.70& 0.26\\
dim2 & -0.46 & -0.84 &-0.3& 1.00 & 0.35 & 0.65\\
\end{array}
\right)
$$ (svd-B)

5. **A new document or query** is first mapped to it's BoW-vector $q$. Then the representation of this vector in the latent semantic space is calculated by 

$$
T'^T \cdot q^T.
$$

For example, if the query consists of the words *astronaut, moon* and *car*, then the corresponding BoW-vector is

$$
q=(0,1,1,1,0)
$$

and the coordinates of this query in the latent semantic space are


$$
T'^T \cdot q^T = 
\left(
\begin{array}{c}
0.13+0.48+0.7 \\
-0.33-0.51+0.35	
\end{array}
\right)
=
\left(
\begin{array}{c}
1.31 \\
-0.49	
\end{array}
\right).
$$

In the figure below, the 6 documents (columns of matrix {eq}`svd-B` ) and the new query are plotted in the latent semantic space. 
<figure align="center">
<img width="400" src="https://maucher.home.hdm-stuttgart.de/Pics/docsQueryIn2dimSpaceLSI.png">
<figcaption>Representations of documents and query vector in latent semantic space</figcaption>
</figure>

As mentioned earlier document-vectors are often normed to unique-length. These normed vectors are shown in the figure below.
<figure align="center">
<img width="400" src="https://maucher.home.hdm-stuttgart.de/Pics/docsQueryNormedIn2dimSpace.png">
<figcaption>Representations of normed documents and normed query vector in latent semantic space</figcaption>
</figure>

As can be seen the documents belonging to the topic *vehicles* are located in an other region of the latent semantic space than the documents, which refer to the topic *space*. Moreover, the query-vector, which contains *space*-words is in the region of *space*-documents.

```python

```
