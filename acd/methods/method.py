from abc import ABC, abstractmethod
from scipy.special import expit as sigmoid
import torch
import numpy as np

# wrapper class for explanation methods

class ExplanationMethod(ABC):
	
	def __init__(self,method,model,inputs,answers):
		self.method = method # this is a string that stores the method name
		self.model = model
		self.inputs = inputs
		self.answers = answers
		super().__init__()

	"""
		Returns np.array
	"""
	@abstractmethod
	def explain(self, batch, start, end):
		pass