import numpy as np

class Config(object):

	bandwidth = 320 #336#544#544#856#

	ftop = 1670 #1732#1088#1711.89550781#1712#1088#1088#1712#

	nchans = 4096 #672

	tsamp = 64e-6  #256*10**-6#0.0004818823529412#6.22056074766355e-05#0.00030624299#0.0004818823529412#0.0004818823529412#0.00030624299##

	min_width = 512 #16384#1024#256

	max_width = 16384 #microseconds

	width_step = 2 #2

	dm_min = 0

	dm_max = 4000

	KDM = 1.0 / 2.41e-4

	window_size = 256

	image_size = 1024

	overlap = 50

	fudge_factor = 50

	num_of_classes = 2

	batch_size = 4

	threshold = 0.1
	
	#weights_directory = "/scratch/nas_spiders/belmonte/PhD_continuation/FRB_Net/FRB_Net_pytorch/parameters/1024_pix_12_sigma_corrected_for_elongation/19.torch"
	
	weights_directory = "/skatvnas3/raghuttam/hermes/passes_in_full_epochs_22.torch"

	resnet_weights_directory = "/skatvnas3/raghuttam/hermes/"#"/raid/scratch/jtian/"#"/scratch/nas_spiders/belmonte/PhD_continuation/FRB_Net/"


	def __init__(self):
		self.fbottom = self.ftop - self.bandwidth
		self.width_array = self.width_step**(np.arange(np.log(self.min_width)/np.log(self.width_step),np.log(self.max_width*self.width_step)/np.log(self.width_step))).astype(int)



