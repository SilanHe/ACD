from .util_objects import Batch, ScoreQueue
from types import *

"""
	agglomerative contextual decomposition
	params:
		batch: Batch -> example
		k: float -> percentile for agglomeration
		method: FunctionType
	returns list of groups, each group is a tuple with the score and the indeces of the group
"""
def get_acd(batch: Batch, k: float, method:  FunctionType):


	# initialize our priority queue and tree
	tree = list()
	score_queue = ScoreQueue()

	len_batch = list(batch.text.size())[0]

	# get unigram scores
	for i in range(len_batch):
		# insert index of word in batch along with its score as (score,index) tuple
		salience = method(batch,i,i)
		score_queue.push([salience,[i],[]])

	# iteratively build up tree
	while not score_queue.isempty():

		selected_groups = score_queue.pop_top_k_percentile(k)
		tree = tree + selected_groups

		# generate new groups of features based on current groups and push them to the priority queue
		for index_selected_group, selected_group in enumerate(selected_groups):
			
			group_start_index = selected_group[1][-1] + 1
			group_end_index = selected_group[1][0] - 1
			
			# find groups that are next to current group in selected_groups
			candidate_groups_index = list()
			for index_candidate_group,candidate_group in enumerate(selected_groups):
				# start and end index of candidate group
				start = candidate_group[1][0]
				end = candidate_group[1][-1]

				if group_start_index >= 0 and group_start_index == start:
					# selected_group + candidate_group

					new_candidate_group_indeces = selected_group[1][:] + candidate_group[1][:]
					candidate_groups_index.append([new_candidate_group_indeces,selected_groups[index_selected_group],selected_groups[index_candidate_group]])

				elif group_end_index < len_batch and group_end_index == end:
					#  candidate_group + selected_group
					new_candidate_group_indeces =  candidate_group[1][:] + selected_group[1][:]
					candidate_groups_index.append([new_candidate_group_indeces,selected_groups[index_candidate_group],selected_groups[index_selected_group]])

			selected_group_score = method(batch,selected_group[1][0],selected_group[1][-1])

			# candidate group contains (index of new group from candidate group and selected_group combo, firstgroup, second group)
			for candidate_group in candidate_groups_index:
				indeces = candidate_group[0]
				candidate_group_score = method(batch,candidate_group[0][0],candidate_group[0][-1])


				score = candidate_group_score - selected_group_score
				score_queue.push([score,indeces,candidate_group[1:]])

	return tree


