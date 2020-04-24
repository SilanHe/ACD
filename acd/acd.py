from util_objects import Batch, ScoreQueue
from types import *

"""
	agglomerative contextual decomposition
	params:
		batch: Batch -> example
		k: float -> percentile for agglomeration
		method: FunctionType
	returns list of groups, each group is a tuple with the score and the indeces of the group
"""
def acd(batch: Batch, k: float, method:  FunctionType):


	# initialize our priority queue and tree
	tree = list()
	score_queue = ScoreQueue()

	len_batch = list(batch.text.size())[0]

	# get unigram scores
	for i in range(len_batch):
		# insert index of word in batch along with its score as (score,index) tuple
		score_queue.add((method(batch,i,i),[i]))

	# iteratively build up tree
	while priority_queue:

		selected_groups = score_queue.pop_top_k_percentile(k)
		tree = tree + selected_groups

		# generate new groups of features based on current groups and add them to the priority queue
		for selected_group in selected_groups:
			
			group_start_index = selected_group[1][-1] + 1
			group_end_index = selected_group[1][0] - 1
			
			# find groups that are next to current group in selected_groups
			candidate_groups_index = list()
			for candidate_group in selected_groups:
				# start and end index of candidate group
				start = candidate_group[1][0]
				end = candidate_group[1][-1]

				if group_start_index >= 0 and group_start_index == start:
					# selected_group + candidate_group

					new_candidate_group_indeces = selected_group[1] + candidate_group[1]
					candidate_groups_index.append(new_candidate_group_indeces)

				elif group_end_index < len_batch and group_end_index == end:
					#  candidate_group + selected_group

					new_candidate_group_indeces =  candidate_group[1] + selected_group[1]
					candidate_groups_index.append(new_candidate_group_indeces)


			selected_group_score = method(batch,selected_group[1][0],selected_group[1][-1])
			for candidate_group in candidate_groups:
				candidate_group_score = method(batch,candidate_group[1][0],candidate_group[1][-1])

				score = candidate_group_score - selected_group_score
				score_queue.add((score,candidate_group))

	return tree


