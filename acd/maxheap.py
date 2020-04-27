# Python3 implementation of Max Heap 

import sys 


# this implementation assumes items are lists or tuples. 
# The absolute value of the first index is considered for the heap manoeuvers.
class MaxHeap: 

	def __init__(self, maxsize): 
		self.maxsize = maxsize 
		self.size = 0
		self.Heap = [[0,None] for i in range(self.maxsize + 1) ]
		self.Heap[0] = [sys.maxsize,None]
		self.FRONT = 1

	# Function to return the position of 
	# parent for the node currently 
	# at pos 
	def parent(self, pos): 
		return pos//2

	# Function to return the position of 
	# the left child for the node currently 
	# at pos 
	def leftChild(self, pos): 
		return 2 * pos 

	# Function to return the position of 
	# the right child for the node currently 
	# at pos 
	def rightChild(self, pos): 
		return (2 * pos) + 1

	# Function that returns true if the passed 
	# node is a leaf node 
	def isLeaf(self, pos): 
		if pos >= (self.size//2) and pos <= self.size: 
			return True
		return False

	# Function to swap two nodes of the heap 
	def swap(self, fpos, spos): 
		self.Heap[fpos], self.Heap[spos] = self.Heap[spos], self.Heap[fpos] 

	# Function to heapify the node at pos 
	def maxHeapify(self, pos): 

		# If the node is a non-leaf node and smaller 
		# than any of its child 
		if not self.isLeaf(pos): 
			if (abs(self.Heap[pos][0]) < abs(self.Heap[self.leftChild(pos)][0]) or
				abs(self.Heap[pos][0]) < abs(self.Heap[self.rightChild(pos)][0])): 

				# Swap with the left child and heapify 
				# the left child 
				if abs(self.Heap[self.leftChild(pos)][0]) > abs(self.Heap[self.rightChild(pos)][0]): 
					self.swap(pos, self.leftChild(pos)) 
					self.maxHeapify(self.leftChild(pos)) 

				# Swap with the right child and heapify 
				# the right child 
				else: 
					self.swap(pos, self.rightChild(pos)) 
					self.maxHeapify(self.rightChild(pos)) 

	# Function to insert a node into the heap 
	def insert(self, element): 
		if self.size >= self.maxsize : 
			return
		self.size+= 1
		self.Heap[self.size] = element 

		current = self.size 

		while abs(self.Heap[current][0]) > abs(self.Heap[self.parent(current)][0]): 
			self.swap(current, self.parent(current)) 
			current = self.parent(current) 

	# Function to print the contents of the heap 
	def Print(self): 
		for i in range(1, (self.size//2)+1): 
			print(" PARENT : "+str(self.Heap[i])+" LEFT CHILD : "+
							str(self.Heap[2 * i])+" RIGHT CHILD : "+
							str(self.Heap[2 * i + 1])) 

	# Function to remove and return the maximum 
	# element from the heap 
	def extractMax(self): 

		popped = self.Heap[self.FRONT] 
		self.Heap[self.FRONT] = self.Heap[self.size] 
		self.size-= 1
		self.maxHeapify(self.FRONT) 
		return popped 

# Driver Code 
if __name__ == "__main__": 
	print('The maxHeap is ') 
	minHeap = MaxHeap(15) 
	minHeap.insert([5,"five"]) 
	minHeap.insert([3,"6"]) 
	minHeap.insert([17,"7"]) 
	minHeap.insert([10,"10"]) 
	minHeap.insert([-84,"856"]) 
	minHeap.insert([19,"32"]) 
	minHeap.insert([6,3421]) 
	minHeap.insert([22,43]) 
	minHeap.insert([9,54]) 

	minHeap.Print() 
	print("The Max val is " + str(minHeap.extractMax())) 

