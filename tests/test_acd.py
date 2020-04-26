import sys
sys.path.append('../')
from acd.acd import acd
from acd.util_objects import Batch
from acd.methods.testmethod import TestMethod
import sst_util
from model_util import get_model
from os.path import join

# snapshot
snapshot_dir = 'models/lstmsentiment'
snapshot_file = join(snapshot_dir, 'best_snapshot_devacc_79.35779571533203_devloss_0.41613781452178955_iter_9000_model.pt')

# get model
sys.path.append('models/lstmsentiment')
model = get_model(snapshot_file)

# get data
inputs, answers, train_iterator, dev_iterator = sst_util.get_sst()

# get sst in tree format
sst_sentences, sst = sst_util.get_sst_PTB("data/trees")
len_sst = len(sst)

for index,tree in enumerate(sst):
	
	sentence = sst_sentences[index]
	
	batch = Batch()
	batch.text = [word.lower() for word in sst_sentences[index]]
	batch.convert_to_long_tensor(inputs)

	test_method = TestMethod(0,0,0,0) # test doesnt use any of these object fields
	tree = acd(batch,0.90,test_method.explain)

	print(tree)

	break

