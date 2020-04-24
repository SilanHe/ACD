from method import ExplanationMethod
import random

class TestMethod(ExplanationMethod):

	def __init__(self,method,model,inputs,answers):
		super().__init__(method,model,inputs,answers)

	def explain(self, batch, start, end):
		random.seed(13)
		
		return sum([random.uniform(-2,2) for i in range(start,end + 1)])

