import nltk
import nltk.corpus
from torchtext import data, datasets
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_sst_PTB(path = "/Users/silanhe/Documents/McGill/Grad/WINTER2020/NLU/acd/tests/data/trees", filesplit = "test"):
	sst_reader = nltk.corpus.BracketParseCorpusReader(path, ".*.txt")
	if filesplit == "test":
		sst_sentences = sst_reader.sents("test.txt")
		sst = sst_reader.parsed_sents("test.txt")
	elif filesplit == "train":
		sst_sentences = sst_reader.sents("train.txt")
		sst = sst_reader.parsed_sents("train.txt")
	elif filesplit == "dev":
		sst_sentences = sst_reader.sents("dev.txt")
		sst = sst_reader.parsed_sents("dev.txt")

	return sst_sentences, sst

# get inputs, answers, training set iterator and dev set iterator
def get_sst():    
	inputs = data.Field(lower='preserve-case')
	answers = data.Field(sequential=False, unk_token=None)

	# build with subtrees so inputs are right
	train_s, dev_s, test_s = datasets.SST.splits(inputs, answers, fine_grained = False, train_subtrees = True,
										   filter_pred=lambda ex: ex.label != 'neutral')
	inputs.build_vocab(train_s, dev_s, test_s)
	answers.build_vocab(train_s)
	
	# rebuild without subtrees to get longer sentences
	train, dev, test = datasets.SST.splits(inputs, answers, fine_grained = False, train_subtrees = False,
									   filter_pred=lambda ex: ex.label != 'neutral')
	
	train_iter, dev_iter, test_iter = data.BucketIterator.splits(
			(train, dev, test), batch_size=1, device=device)

	return inputs, answers, train_iter, dev_iter

# gets the batches of the specified dset, by default 'train'
# batch_nums is a list of int, each of which represent an index you wish to retrieve
# train_iterator is our iterator from get_sst()
# dev_iterator is our iterator from get_sst()
def get_batches(batch_nums, train_iterator, dev_iterator, dset='train'):
	print('getting batches...')
	np.random.seed(13)
	random.seed(13)
	
	# pick data_iterator
	if dset=='train':
		data_iterator = train_iterator
	elif dset=='dev':
		data_iterator = dev_iterator
	
	# actually get batches
	num = 0
	batches = {}
	data_iterator.init_epoch() 
	for batch_idx, batch in enumerate(data_iterator):
		if batch_idx == batch_nums[num]:
			batches[batch_idx] = batch
			num +=1 

		if num == max(batch_nums):
			break
		elif num == len(batch_nums):
			print('found them all')
			break
	return batches

# gets the batches from data_iterator, overloaded version of above function
def get_batches_iterator(batch_nums, data_iterator):
	print('getting batches...')
	np.random.seed(13)
	random.seed(13)
	
	# actually get batches that match indices in batch_nums
	num = 0
	batches = {}
	data_iterator.init_epoch() 
	for batch_idx, batch in enumerate(data_iterator):
		if batch_idx == batch_nums[num]:
			batches[batch_idx] = batch
			num +=1 

		if num == max(batch_nums):
			break
		elif num == len(batch_nums):
			print('found them all')
			break
	return batches