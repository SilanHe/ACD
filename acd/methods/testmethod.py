from .method import ExplanationMethod
import random

class TestMethod(ExplanationMethod):

	def __init__(self,method,model,inputs,answers):
		super().__init__(method,model,inputs,answers)
		random.seed(13)

	def explain(self, batch, start, end):
		
		return sum([random.uniform(-2,2) for i in range(start,end + 1)])

