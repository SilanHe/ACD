import torch

# list of all models
import models.lstmsentiment

# get model
def get_model(snapshot_file):
	print('loading', snapshot_file)
	try:  # load onto gpu
		model = torch.load(snapshot_file)
		print('loaded onto gpu...')
	except:  # load onto cpu
		model = torch.load(snapshot_file, map_location=lambda storage, loc: storage)
		print('loaded onto cpu...')
	return model