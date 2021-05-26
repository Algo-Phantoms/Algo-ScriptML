# APRIORI ALGORITHM

## Introduction

Apriori algorithm was given by R. Agrawal and R. Srikant in 1994 for finding frequent itemsets in a dataset for boolean association rule. The algorithm is called so because it uses prior knowledge of frequent itemset properties. With the help of the association rule, it determines how strongly or how weakly two objects are connected. This algorithm uses a breadth-first search and Hash Tree to calculate the itemset associations efficiently. It is the iterative process for finding the frequent itemsets from the large dataset.

To improve the efficiency of level-wise generation of frequent itemsets, **Apriori Property** is used. It helps in reducing the search space.

Frequent itemsets: Frequent itemsets are those items whose support is greater than the threshold value or user-specified minimum support. It means if A & B are the frequent itemsets together, then individually A and B should also be the frequent itemset. Suppose there are the two transactions: A= {1,2,3,4,5}, and B= {2,3,7}, in these two transactions, 2 and 3 are the frequent itemsets.

![](https://djinit-ai.github.io/images/Apriori-Algorithm-2.png)

## Apriori Property

According to this property, all subsets of a frequent itemset must also be frequent.

## Steps for Apriori Algorithm

Step 1: Determine the support of itemsets in the transactional database, and select the minimum support and confidence.

Step 2: Take all supports in the transaction with higher support value than the minimum or selected support value.

Step 3: Find all the rules of these subsets that have higher confidence value than the threshold or minimum confidence.

Step 4: Sort the rules as the decreasing order of lift.

## Advantages of Apriori Algorithm

▪ This is an easy to understand algorithm.

▪ The join and prune steps of the algorithm can be easily implemented on large datasets.

## Disadvantages of Apriori Algorithm

▪ The apriori algorithm works slowly as compared to other algorithms.

▪ The overall performance can be reduced as it scans the database for multiple times.

▪ The time complexity and space complexity of the apriori algorithm is O(2D), which is very high. Here D represents the horizontal width present in the database.

## References

▪ https://www.geeksforgeeks.org/apriori-algorithm/

▪ https://www.javatpoint.com/apriori-algorithm-in-machine-learning
