# *Genetic Algorithm*


##  What is Genetic Algorithm ?

Genetic Algorithm (GA) is a search-based optimization technique based on the principles of **Genetics and Natural Selection**. These are intelligent exploitation of random search provided with historical data to direct the search into the region of better performance in solution space. **They are commonly used to generate high-quality solutions for optimization problems and search problems.** In simpler terms GA is heavily inspired by the process of natural selection to find the best solution to a problem.

## Main Characteristics of GA are :

- The genetic algorithm works with a coding of the parameter set, not the parameters themselves.
- The genetic algorithm initiates its search from a population of points, not a single point.
- The genetic algorithm uses payoff information, not derivatives.
- The genetic algorithm uses probabilistic transition rules, not deterministic ones.

#  Block Diagram of GA :

![ga](https://user-images.githubusercontent.com/63098466/119253739-39aa8400-bbd0-11eb-9770-2bc5f86a712e.JPG)

## Important Parameters - 
**Fitness Assignment**
- Fitness score is the number of characters which differ from characters in target string at a particular index. So individual having lower fitness value is given more preference.  

**Selection**
- The idea is to give preference to the individuals with good fitness scores and allow them to pass there genes to the successive generations.

**Crossover**
- After selection operation, simple crossover proceeds. The main objective of crossover is to reorganize the information of two different individuals and produce a new one. This represents mating between individuals. Two individuals are selected using selection operator and crossover sites are chosen randomly. Then the genes at these crossover sites are exchanged thus creating a completely new individual (offspring).

**Mutation**
- Mutation is a background operator which protects against some irrecoverable loss. It is an occasional random alteration of the value in the string position. Mutation is needed because even though reproduction and crossover effectively search and recombine extent notions, occasionally, they may lose some potentially useful genetic material.
##  Advantage of GA :

GAs have various advantages which have made them immensely popular. These include : 

-   Does not require any derivative information (which may not be available for many real-world problems).  
-   Is faster and more efficient as compared to the traditional methods.  
-   Has very good parallel capabilities. 
-   Optimizes both continuous and discrete functions and also multi-objective problems.    
-   Provides a list of “good” solutions and not just a single solution.   
-   Always gets an answer to the problem, which gets better over the time.  
-   Useful when the search space is very large and there are a large number of parameters involved.

## Drawbacks of GA :
Like any technique, GAs also suffer from a few limitations. These include −

-   GAs are not suited for all problems, especially problems which are simple and for which derivative information is available. 
-   Fitness value is calculated repeatedly which might be computationally expensive for some problems.    
-   Being stochastic, there are no guarantees on the optimality or the quality of the solution.  
-   If not implemented properly, the GA may not converge to the optimal solution.
