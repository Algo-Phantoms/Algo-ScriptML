# FP GROWTH ALGORITHM

## Introduction

FP growth algorithm or the Frequent Pattern Growth Algorithm is an improvement of apriori algorithm. The two primary drawbacks of the Apriori Algorithm were that at each step, candidate sets had to be rebuilt and to build those, the algorithm had to repeatedly scan the database. These properties in turn made the algorithm slower. FP algorithm overcomes the disadvantages of the Apriori algorithm by storing all the transactions in a Tree Data Structure.

## FP Tree

FP tree is the core concept of the whole FP Growth algorithm. It is the compressed representation of the itemset database. The tree structure not only reserves the itemset in DB but also keeps track of the association between itemsets. The tree is constructed by taking each itemset and mapping it to a path in the tree one at a time. The whole idea behind this construction is that more frequently occurring items will have better chances of sharing items. We then mine the tree recursively to get the frequent pattern. Pattern growth, the name of the algorithm, is achieved by concatenating the frequent pattern generated from the conditional FP trees.

A FP tree looks something like this:

![](https://miro.medium.com/max/875/1*P5CAJ1_b89rO09e6hFkWKA.png)

## Advantages

▪ This algorithm needs to scan the database only twice when compared to Apriori which scans the transactions for each iteration.

▪ The pairing of items is not done in this algorithm and this makes it faster.

▪ The database is stored in a compact version in memory.

▪ It is efficient and scalable for mining both long and short frequent patterns.

## Disadvantages

▪ FP Tree is more cumbersome and difficult to build than Apriori.

▪ It may be expensive.

▪ When the database is large, the algorithm may not fit in the shared memory.

## References

▪ https://www.geeksforgeeks.org/ml-frequent-pattern-growth-algorithm/

▪ https://www.softwaretestinghelp.com/fp-growth-algorithm-data-mining/

▪ https://towardsdatascience.com/fp-growth-frequent-pattern-generation-in-data-mining-with-python-implementation-244e561ab1c3
