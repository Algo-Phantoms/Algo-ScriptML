# Principal Component Analysis-PCA
## Introduction
### Definition
It is considered to be one of the most used unsupervised algorithms and can be seen as the most popular dimensionality reduction algorithm as it is used to minimize the dimensionality of large dataset
while preserving as much information as possible. The goal of PCA is to identify and detect the corelation between variables.
### History
The PCA was founded by **KARL PEARSON** in 1901 as an analogue of a major axis theorem in technology; it was also expanded differently and was renamed **HAROLD HOTELLING** in 1930.
 ![](https://ashutoshtripathicom.files.wordpress.com/2019/07/pca_title.jpg)
## How PCA Alogorithm works?
   * Standardize the data. It standardize the stage for the first continuous variance such as each of them contributing equally to the analysis.
     
     ![](https://builtin.com/sites/default/files/styles/ckeditor_optimize/public/inline-images/Principal%20Component%20Analysis%20Standardization.png)

     Once the standardization is done, all the variables will be transformed to the same scale.
   * Calculate the covariance matrix of elements from the database.If we take a 2-dimensional database, this will lead to a 2x2 Covariance matrix.
   * Find the Eigenvectors and Eigenvalues from the covariance matrix or corelation matrix, or perform Singular Vector Decomposition.
  We will take a square matrix. _**ƛ**_ is an eigenvalue for a matrix **A** if it is a solution of the characteristic equation:
                     **det( ƛI - A ) = 0**
                      
   _**I**_ is the identity matrix of the same dimension as **A** which is a required condition for the matrix subtraction as well in this case and **det** is the determinant of the matrix. For each eigenvalue **ƛ**, a corresponding eigen-vector v, can be found by solving
                      **ƛI - A )v = 0**
    ![](https://builtin.com/sites/default/files/styles/ckeditor_optimize/public/inline-images/Principal%20Component%20Analysis%20Principal%20Components.png)
   * Sort Eigenvalues in descending order and choose the _**k**_ eigenvectors that correspond to the _**k**_ largest eigenvalues where _**k**_ is the number of dimensions of the new feature subspace (k<=d).
   * Create the projection matrix **W** from the selected *k* eigenvectors.
   * Change the original dataset **X** via **W** to obtain a k-dimensional feature subspace **Y**

## Advantages
   * **Deleting Related Features** : After applying PCA to your database, all Principal Components are independent. There is no reunion between them.
   * **Reduce excess** : Excessive delays occur especially when there is too much variation in the database. Therefore, PCA helps to overcome the problem of transmission by reducing the number of factors
   * **Improving visualization** : It is very difficult to visualize and understand the data in high magnitude. PCA converts high-resolution data into low-level data (size 2) for easy reference. We can use 2D Scree Plot to see which key elements lead to higher fragmentation and have a greater impact compared to other key elements.
   * **Improves Algorithm Performance** : With so many features, the performance of your algorithm will be greatly reduced. PCA is the most common way to speed up your machine learning algorithm by removing the corresponding dynamics that do not contribute to decision-making. The training time for algorithms is greatly reduced by a small number of features

## Disadvantages
   * **Data Loss** : Although the Main components attempt to cover the high variability between the elements in the database, if we do not select the number of Main elements with care, it may lose some information compared to the original list of features.
   * **Independent variables have been slightly interpreted** : After applying PCA to the dataset, your original features will become the Main Topics. Key elements are a direct combination of your personal features. Key elements are unreadable and are not interpreted as actual elements.
   * **Data setting must be prior to PCA** : We must set up your details before using the PCA, otherwise the PCA will not be able to find the correct Key Features.

## Applications
   * PCA is widely used as a process to **reduce the size of domains** such as **face recognition**, **computer vision**, **noise filtering** and **image compression**. 
   * It is also used to **find patterns in high-profile data** in the field of **finance**, **data mining**, **bioinformatics**, **psychology**, etc.
   * Gene data Analysis
   
## Reading References
   * https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
   * https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
   * https://setosa.io/ev/principal-component-analysis/
## Video References
   * https://www.youtube.com/watch?v=2NEu9dbM4A8
   * https://www.youtube.com/watch?v=n7npKX5zIWI
   * https://www.youtube.com/watch?v=uFbDWu0tDrE
  
  
