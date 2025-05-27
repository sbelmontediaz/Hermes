import numpy as np
import yaml
import os

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


	def __init__(self,config_file=None):
		if config_file is None or not os.path.exists(config_file):
			raise FileNotFoundError(f"Config file {config_file} does not exist.")
		# Load the config file
		with open(config_file) as f:
			config = yaml.safe_load(f)
		self.bandwidth = config['bandwidth'] if 'bandwidth' in config else self.bandwidth
		self.ftop = config['ftop'] if 'ftop' in config else self.ftop
		self.nchans = config['nchans'] if 'nchans' in config else self.nchans
		self.tsamp = config['tsamp'] if 'tsamp' in config else self.tsamp
		self.min_width = config['min_width'] if 'min_width' in config else self.min_width
		self.max_width = config['max_width'] if 'max_width' in config else self.max_width
		self.width_step = config['width_step'] if 'width_step' in config else self.width_step
		self.dm_min = config['dm_min'] if 'dm_min' in config else self.dm_min
		self.dm_max = config['dm_max'] if 'dm_max' in config else self.dm_max
		self.KDM = config['KDM'] if 'KDM' in config else self.KDM
		self.window_size = config['window_size'] if 'window_size' in config else self.window_size
		self.image_size = config['image_size'] if 'image_size' in config else self.image_size
		self.overlap = config['overlap'] if 'overlap' in config else self.overlap
		self.fudge_factor = config['fudge_factor'] if 'fudge_factor' in config else self.fudge_factor
		self.num_of_classes = config['num_of_classes'] if 'num_of_classes' in config else self.num_of_classes
		self.batch_size = config['batch_size'] if 'batch_size' in config else self.batch_size
		self.threshold = config['threshold'] if 'threshold' in config else self.threshold
		self.weights_directory = config['weights_directory'] if 'weights_directory' in config else self.weights_directory
		self.resnet_weights_directory = config['resnet_weights_directory'] if 'resnet_weights_directory' in config else self.resnet_weights_directory
		self.clustering = config['clustering'] if 'clustering' in config else False
		
		self.fbottom = self.ftop - self.bandwidth
		self.width_array = self.width_step**(np.arange(np.log(self.min_width)/np.log(self.width_step),np.log(self.max_width*self.width_step)/np.log(self.width_step))).astype(int)



