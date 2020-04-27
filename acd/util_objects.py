from .maxheap import MaxHeap
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Batch:

	def __init__(self):
		self.text = None
		self.label = None

	def convert_to_long_tensor(self, inputs):
		# convert batch.text to long tensor where each long is a word according to inputs vocab
		vector = [[inputs.vocab.stoi[word]] for word in self.text]
		word_tensor = torch.LongTensor(vector).to(device)
		self.text = word_tensor
		return word_tensor

class ScoreQueue():

	def __init__(self):
		self.pq = MaxHeap(100) # 1000 should not ever be reached

	def pop_top_k_percentile(self,k: float):

		# get list of magnitudes
		# assume each item in self.pq is a tuple
		len_pq = self.pq.size
		num_top_k_percentile = max(len_pq - int(k * len_pq),2)

		pops = list()
		for i in range(num_top_k_percentile):
			pops.append(self.pop())

		return pops

	def push(self,item):
		self.pq.insert(item)

	def pop(self):
		return self.pq.extractMax()

	def isempty(self):
		return self.pq.size == 0
