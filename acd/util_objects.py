import heapq

class Batch:

	def __init__(self):
		self.text = None
		self.label = None

class ScoreQueue():

	def __init__(self):
		self.pq = list()

	def pop_top_k_percentile(k: float):

		# get list of magnitudes
		# assume each item in self.pq is a tuple
		len_pq = len(pq)
		num_top_k_percentile = len_pq - int(k * len_pq)

		pops = list()
		for i in range(num_top_k_percentile):
			pops.append(self.pop())

		return pops

	def add(item):
		heappush(self.pq,item)

	def pop(item):
		heappop(self.pq,item)

