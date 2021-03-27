class C5decisionTree:
	"""Binary tree implementation with true and false branch. """
	def __init__(self, col=-1, value=None, trueBranch=None, falseBranch=None, results=None):
		self.col = col
		self.value = value
		self.trueBranch = trueBranch
		self.falseBranch = falseBranch
		self.results = results # None for nodes, not None for leaves

def divideSet(rows, column, value):
	splittingFunction = None
	if isinstance(value, int) or isinstance(value, float): # for int and float values
		splittingFunction = lambda row : row[column] >= value
	else: # for strings 
		splittingFunction = lambda row : row[column] == value
	list1 = [row for row in rows if splittingFunction(row)]
	list2 = [row for row in rows if not splittingFunction(row)]
	return (list1, list2)


def uniqueCounts(rows):
	results = {}
	for row in rows:
		r = row[-1]
		if r not in results: results[r] = 0
		results[r] += 1
	return results
    
def entropy(rows):
	from math import log
	log2 = lambda x: log(x)/log(2)
	results = uniqueCounts(rows)

	entr = 0.0
	for r in results:
		p = float(results[r])/len(rows)
		entr -= p*log2(p)
	return entr

def growDecisionTreeFrom(rows, evaluationFunction=entropy):
	"""Grows and then returns a binary decision tree. 
	evaluationFunction: entropy""" 

	if len(rows) == 0: return DecisionTree()
	currentScore = evaluationFunction(rows)

	bestGain = 0.0
	bestAttribute = None
	bestSets = None

	columnCount = len(rows[0]) - 1  # last column is the result/target column
	for col in range(0, columnCount):
		columnValues = [row[col] for row in rows]

		for value in columnValues:
			(set1, set2) = divideSet(rows, col, value)

			# Gain -- Entropy
			p = float(len(set1)) / len(rows)
			gain = currentScore - p*evaluationFunction(set1) - (1-p)*evaluationFunction(set2)
			if gain>bestGain and len(set1)>0 and len(set2)>0:
				bestGain = gain
				bestAttribute = (col, value)
				bestSets = (set1, set2)

	if bestGain > 0:
		trueBranch = growDecisionTreeFrom(bestSets[0])
		falseBranch = growDecisionTreeFrom(bestSets[1])
		return C5decisionTree(col=bestAttribute[0], value=bestAttribute[1], trueBranch=trueBranch, falseBranch=falseBranch)
	else:
		return C5decisionTree(results=uniqueCounts(rows))

def prune(tree, minGain, evaluationFunction=entropy, notify=False):
	"""Prunes the obtained tree according to the minimal gain (entropy). """
	# recursive call for each branch
	if tree.trueBranch.results == None: prune(tree.trueBranch, minGain, evaluationFunction, notify)
	if tree.falseBranch.results == None: prune(tree.falseBranch, minGain, evaluationFunction, notify)

	# merge leaves (potentionally)
	if tree.trueBranch.results != None and tree.falseBranch.results != None:
		tb, fb = [], []

		for v, c in tree.trueBranch.results.items(): tb += [[v]] * c
		for v, c in tree.falseBranch.results.items(): fb += [[v]] * c

		p = float(len(tb)) / len(tb + fb)
		delta = evaluationFunction(tb+fb) - p*evaluationFunction(tb) - (1-p)*evaluationFunction(fb)
		if delta < minGain:	
			if notify: print('A branch was pruned: gain = %f' % delta)		
			tree.trueBranch, tree.falseBranch = None, None
			tree.results = uniqueCounts(tb + fb)

def classify(observations, tree):
		if tree.results != None:  # leaf
			return tree.results
		else:
			v = observations[tree.col]
			branch = None
			if isinstance(v, int) or isinstance(v, float):
				if v >= tree.value: branch = tree.trueBranch
				else: branch = tree.falseBranch
			else:
				if v == tree.value: branch = tree.trueBranch
				else: branch = tree.falseBranch
		return classify(observations, branch)





