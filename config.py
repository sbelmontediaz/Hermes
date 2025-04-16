import numpy as np

class Config(object):

	bandwidth = 0

	ftop = 0

	nchans = 0

	tsamp = 0

	min_width = 512#16384#1024#256

	max_width = 16384#4096#8192#

	width_step = 2

	dm_min = 0

	dm_max = 8000

	KDM = 1.0 / 2.41e-4

	window_size = 256

	image_size = 1024

	overlap = 50

	fudge_factor = 50

	num_of_classes = 2

	batch_size = 4

	threshold = 0.1
	
	#weights_directory = "/scratch/nas_spiders/belmonte/PhD_continuation/FRB_Net/FRB_Net_pytorch/parameters/1024_pix_12_sigma_corrected_for_elongation/19.torch"
	
	weights_directory = "/scratch/nas_spiders/belmonte/PhD_continuation/FRB_Net/FRB_Net_pytorch/parameters/50/passes_in_full_epochs_22.torch"

	resnet_weights_directory = "/scratch/nas_spiders/belmonte/PhD_continuation/FRB_Net/"#"/raid/scratch/jtian/"#"/scratch/nas_spiders/belmonte/PhD_continuation/FRB_Net/"


	def __init__(self):
		self.fbottom = self.ftop - self.bandwidth
		self.width_array = self.width_step**(np.arange(np.log(self.min_width)/np.log(self.width_step),np.log(self.max_width*self.width_step)/np.log(self.width_step))).astype(int)



