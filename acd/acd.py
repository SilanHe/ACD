from util_objects import Batch, Tree, ScoreQueue
from types import *

"""
	agglomerative contextual decomposition
	params:
		batch: Batch -> example
		k: float -> percentile for agglomeration
		method: FunctionType
	returns Tree
"""
def acd(batch: Batch, k: float, method:  FunctionType):

	# initialize our priority queue and tree
	tree = Tree()
	score_queue = ScoreQueue()

	len_batch = list(batch.text.size())[0]
	scores_batch = list()
	for i in range(len_batch):
		# insert index of word in batch along with its score as (score,index) tuple
		scores_batch.append(method(batch,i,i))

	# iteratively build up tree
	while priority_queue:

		selectedGroups = score_queue.poptopkpercentile(k)
		tree.add(selectedGroups)



