from .method import ExplanationMethod

class CD(ExplanationMethod):

	def __init__(self,method,model,inputs,answers):
		super().__init__(method,model,inputs,answers)

	def explain(self, batch, start, end):

		def decomp_three(a, b, c, activation):
			a_contrib = 0.5 * (activation(a + c) - activation(c) + activation(a + b + c) - activation(b + c))
			b_contrib = 0.5 * (activation(b + c) - activation(c) + activation(a + b + c) - activation(a + c))
			return a_contrib, b_contrib, activation(c)

		def decomp_tanh_two(a, b):
			return 0.5 * (np.tanh(a) + (np.tanh(a + b) - np.tanh(b))), 0.5 * (np.tanh(b) + (np.tanh(a + b) - np.tanh(a)))
        
		def CD(batch, model, start, stop):
			weights = model.lstm.state_dict()

			# Index one = word vector (i) or hidden state (h), index two = gate
			W_ii, W_if, W_ig, W_io = np.split(weights['weight_ih_l0'].cpu(), 4, 0)
			W_hi, W_hf, W_hg, W_ho = np.split(weights['weight_hh_l0'].cpu(), 4, 0)
			b_i, b_f, b_g, b_o = np.split(weights['bias_ih_l0'].cpu().numpy() + weights['bias_hh_l0'].cpu().numpy(), 4)
			
			word_vecs = model.embed(batch.text)[:,0].data

			T = word_vecs.size(0)
			word_vecs = [word_vec.cpu() for word_vec in word_vecs]
			relevant = np.zeros((T, model.hidden_dim))
			irrelevant = np.zeros((T, model.hidden_dim))
			relevant_h = np.zeros((T, model.hidden_dim))
			irrelevant_h = np.zeros((T, model.hidden_dim))

			for i in range(T):
				if i > 0:
					prev_rel_h = relevant_h[i - 1]
					prev_irrel_h = irrelevant_h[i - 1]
				else:
					prev_rel_h = np.zeros(model.hidden_dim)
					prev_irrel_h = np.zeros(model.hidden_dim)

				rel_i = np.dot(W_hi, prev_rel_h)
				rel_g = np.dot(W_hg, prev_rel_h)
				rel_f = np.dot(W_hf, prev_rel_h)
				rel_o = np.dot(W_ho, prev_rel_h)
				irrel_i = np.dot(W_hi, prev_irrel_h)
				irrel_g = np.dot(W_hg, prev_irrel_h)
				irrel_f = np.dot(W_hf, prev_irrel_h)
				irrel_o = np.dot(W_ho, prev_irrel_h)

				if i >= start and i <= stop:
					rel_i = rel_i + np.dot(W_ii, word_vecs[i])
					rel_g = rel_g + np.dot(W_ig, word_vecs[i])
					rel_f = rel_f + np.dot(W_if, word_vecs[i])
					rel_o = rel_o + np.dot(W_io, word_vecs[i])
				else:
					irrel_i = irrel_i + np.dot(W_ii, word_vecs[i])
					irrel_g = irrel_g + np.dot(W_ig, word_vecs[i])
					irrel_f = irrel_f + np.dot(W_if, word_vecs[i])
					irrel_o = irrel_o + np.dot(W_io, word_vecs[i])

				rel_contrib_i, irrel_contrib_i, bias_contrib_i = decomp_three(rel_i, irrel_i, b_i, sigmoid)
				rel_contrib_g, irrel_contrib_g, bias_contrib_g = decomp_three(rel_g, irrel_g, b_g, np.tanh)

				relevant[i] = rel_contrib_i * (rel_contrib_g + bias_contrib_g) + bias_contrib_i * rel_contrib_g
				irrelevant[i] = irrel_contrib_i * (rel_contrib_g + irrel_contrib_g + bias_contrib_g) + (rel_contrib_i + bias_contrib_i) * irrel_contrib_g

				if i >= start and i < stop:
					relevant[i] += bias_contrib_i * bias_contrib_g
				else:
					irrelevant[i] += bias_contrib_i * bias_contrib_g

				if i > 0:
					rel_contrib_f, irrel_contrib_f, bias_contrib_f = decomp_three(rel_f, irrel_f, b_f, sigmoid)
					relevant[i] += (rel_contrib_f + bias_contrib_f) * relevant[i - 1]
					irrelevant[i] += (rel_contrib_f + irrel_contrib_f + bias_contrib_f) * irrelevant[i - 1] + irrel_contrib_f * relevant[i - 1]

				o = sigmoid(np.dot(W_io, word_vecs[i]) + np.dot(W_ho, prev_rel_h + prev_irrel_h) + b_o)
				rel_contrib_o, irrel_contrib_o, bias_contrib_o = decomp_three(rel_o, irrel_o, b_o, sigmoid)
				new_rel_h, new_irrel_h = decomp_tanh_two(relevant[i], irrelevant[i])
				#relevant_h[i] = new_rel_h * (rel_contrib_o + bias_contrib_o)
				#irrelevant_h[i] = new_rel_h * (irrel_contrib_o) + new_irrel_h * (rel_contrib_o + irrel_contrib_o + bias_contrib_o)
				relevant_h[i] = o * new_rel_h
				irrelevant_h[i] = o * new_irrel_h

			W_out = model.hidden_to_label.weight.data.cpu()
			
			# Sanity check: scores + irrel_scores should equal the LSTM's output minus model.hidden_to_label.bias
			scores = np.dot(W_out, relevant_h[T - 1])
			irrel_scores = np.dot(W_out, irrelevant_h[T - 1])

			return scores, irrel_scores

		score, _ = CD(batch, self.model, start, stop)

		return score

		