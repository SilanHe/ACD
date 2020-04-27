import sys
sys.path.append('../')
import acd
import sst_util
from model_util import get_model
from os.path import join

def main():
	# snapshot
	snapshot_dir = './models/lstmsentiment'
	snapshot_file = join(snapshot_dir, 'best_snapshot_devacc_79.35779571533203_devloss_0.41613781452178955_iter_9000_model.pt')

	# get model
	sys.path.append(snapshot_dir)
	model = get_model(snapshot_file)

	# get data
	inputs, answers, train_iterator, dev_iterator = sst_util.get_sst()

	# get sst in tree format
	sst_sentences, sst = sst_util.get_sst_PTB("data/trees")
	len_sst = len(sst)

	count = 0
	for index,tree in enumerate(sst):
		
		sentence = ' '.join(sst_sentences[index])
		
		batch = acd.Batch()
		batch.text = [word.lower() for word in sst_sentences[index]]
		batch.convert_to_long_tensor(inputs)
		try:
			test_method = acd.CD("cd",model,inputs,answers) # test doesnt use any of these object fields
			tree = acd.get_acd(batch,0.80,test_method.explain)
			print(index, sentence,'\n',tree)
		except Exception as e:
			print(e)
		

		count += 1
		if count > 10:
			break
main()
