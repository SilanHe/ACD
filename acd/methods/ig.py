from .method import ExplanationMethod

class IG(ExplanationMethod):

	def __init__(self,method,model,inputs,answers):
		super().__init__(method,model,inputs,answers)

	def explain(self, batch, start, end): 
		
		def ig(batch, model, inputs, answers):

			x = model.embed(batch.text)
			T = x.size(0)
			word_vecs = [word_vec.cpu() for word_vec in x]

			x_dash = torch.zeros_like(x)
			sum_grad = None
			grad_array = None
			x_array = None

			# get Predicted label
			with torch.no_grad():
				model.eval()
				pred=torch.argmax(model(x))
			model.train()

			# ig
			for k in range(T):
				model.zero_grad()
				step_input = x_dash + k * (x - x_dash) / T
				step_output = model(step_input)
				step_pred = torch.argmax(step_output)
				step_grad = torch.autograd.grad(step_output[0][pred.item()], x)[0]
				if sum_grad is None:
					sum_grad = step_grad
					grad_array = step_grad
					x_array = step_input
				else:
					sum_grad += step_grad
					grad_array = torch.cat([grad_array, step_grad])
					x_array = torch.cat([x_array, step_input])

			sum_grad = sum_grad / T
			sum_grad = sum_grad * (x - x_dash)
			sum_grad = sum_grad.sum(dim=2)

			relevances = sum_grad.detach().cpu().numpy()
			relevances = np.round(np.reshape(relevances,T),3)
			
			return answers.vocab.itos[pred], relevances

		_, scores = ig(batch, self.model, self.inputs, self.answers)

		score = np.sum(scores[start:end+1])
		
		return score
