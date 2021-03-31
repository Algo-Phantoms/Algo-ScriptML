from __future__ import print_function
import pandas as pd
import numpy as np
# Some Global utility functions :

# for finding unique values in a feature


def unique_vals(rows, col):
    return set([row[col] for row in rows])

# Counting the number of each kinf od exmple


def class_counts(rows):
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)

# ----------------------------------------------------------------------------------------------------------------------------------

# Class for framing a 'Question' , hoose answer is either true or false based on the match funtion which compares
# the value of example at that column with the data


class Question:

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))


# Comparing the information gain to find the best split or the node which is most valuable at a point
def find_best_split(rows):
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1  # number of columns

    for col in range(n_features):  # for each feature

        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:  # for each value

            question = Question(col, val)

            # try splitting the dataset
            true_rows, false_rows = partition(rows, question)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question

# diving the data into true_rorwsa nd false_rows based on Question


def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

# Recursive function to build Decision tree using C.A.R.T algorithm


def build_tree(rows):

    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    gain, question = find_best_split(rows)

    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain == 0:
        return Leaf(rows)

    # If we reach here, we have found a useful feature / value
    # to partition on.
    true_rows, false_rows = partition(rows, question)

    # Recursively build the true branch.
    true_branch = build_tree(true_rows)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # dependingo on the answer.
    return Decision_Node(question, true_branch, false_branch)


def gini(rows):
    # calculating gini impurity
    # Source: https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity

    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity

# calculating I.G.


def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)

# leaf class contains the final predictions or label values


class Leaf:
    def __init__(self, rows):
        self.predictions = class_counts(rows)

# Decision node is the node at whcih question is asked


class Decision_Node:
    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


# printing the tree
def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print(spacing + str(node.question))

    # Call this function recursively on the true branch
    print(spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print(spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")

# for testing how the tree performed


def classify(row, node):
    """See the 'rules of recursion' above."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


def print_leaf(counts):
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs


if __name__ == "__main__":

    training_data = [
        ['Sunny', 'Hot', 'Weak', 'No'],
        ['Sunny', 'Hot', 'Strong', 'No'],
        ['Sunny', 'Cool', 'Weak', 'No'],
        ['Overcast', 'Hot', 'Weak', 'Yes'],
        ['Overcast', 'Cool', 'Strong', 'Yes'],
        ['Overcast', 'Hot', 'Weak', 'Yes'],
        ['Rain', 'Mild', 'Weak', 'Yes'],
        ['Rain', 'Cool', 'Strong', 'No'],
        ['Rain', 'Mild', 'Strong', 'No']
    ]
    header = ["Outlook", "temperature", "Wind", "Play-tennis ?"]
    print("The PLay Tennis data-set (categorical):", '\n')

    arr = training_data
    df = pd.DataFrame(training_data, columns=header)

    print(df, '\n')

    testing_data = [
        ['Rain', 'Hot', 'Strong', 'No'],
        ['Overcast', 'Hot', 'Strong', 'Yes'],
        ['Sunny', 'Cool', 'Weak', 'Yes'],
        ['Overcast', 'Hot', 'Weak', 'Yes'],
        ['Rain', 'Cool', 'Strong', 'No'],
        ['Overcast', 'Hot', 'Weak', 'Yes'],
        ['Rain', 'Mild', 'Weak', 'Yes'],
        ['Sunny', 'Hot', 'weak', 'Yes'],
        ['Sunny', 'Hot', 'Strong', 'No']
    ]

    header = ["Outlook", "temperature", "Wind", "Play-tennis ?"]

    my_tree = build_tree(training_data)
    print('\n', 'The Tree Structure :', '\n')

    print_tree(my_tree)
    print('\n',)

    print("Results : ", '\n')
    for row in testing_data:
        print("Actual: %s. Predicted: %s" %
              (row[-1], print_leaf(classify(row, my_tree))))
    print('\n')

    # Some numerical data for testing purpose :
    training_data = [
        [5.1, 3.5, 1.4, 0.2, 'setosa'],
        [4.9, 3.0, 1.4, 0.2, 'setosa'],
        [5.1,	3.8,	1.6,	0.2, 'setosa'],
        [7.0,	3.2,	4.7,	1.4,	'versicolor'],
        [6.6,	2.9	, 4.6,	1.3,	'versicolor'],
        [5.2,	2.7,	3.9,	1.4,	'versicolor'],
        [7.7,	3.8,	6.7,	2.2,	'virginica'],
        [6.0,	2.2,	5.0,	1.5,	'virginica'],
        [6.3,	2.7,	4.9,	1.8,	'virginica']
    ]

    testing_data = [
        [4.9,	3.1,	1.5,	0.1,	'setosa'],
        [4.8,	3.4,	1.6,	0.2,	'setosa'],
        [6.4,	3.2,	4.5,	1.5,	'versicolor'],
        [4.9,	3.1,	1.5,	0.1,	'setosa'],
        [6.9,	3.1,	5.1,	2.3,	'virginica'],
        [5.5,	2.5,	4.0,	1.3,	'versicolor'],
        [5.9,	3.0,	5.1,	1.8,	'virginica'],
        [6.0,	2.2,	5.0,	1.5,	'virginica'],
        [5.5,	2.4,	3.8,	1.1,	'versicolor']
    ]

    header = ["sepal_length", "sepal_width",
              "petal_length", "petal_width",	"species"]
