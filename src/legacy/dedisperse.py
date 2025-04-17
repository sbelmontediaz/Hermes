import logging
import numpy as np
import matplotlib.pyplot as plt
import sigpyproc,sigpyproc.readers
import sys,os
import math
import h5py
import glob
import time
import argparse
from ddplan import ddplan
from config import Config
from scipy.signal import savgol_filter
from astropy.stats import sigma_clipped_stats
import numbers
from numpy.lib.stride_tricks import as_strided
from skimage.measure import block_reduce
from pathlib import Path
from ddplan import ddplan
import utils
from py_astro_accelerate import *
import ctypes

#Object to manage the dedispersion of files.
class Dedispersing_files(object):
	def __init__(self,config, filename, output_directory, subbanded, nsubint=100, no_rfi_cleaning=False, no_zdot=False, use_astro_accelerate=False):
		self.config 						=		config
		self.filename 						=		Path(filename)				#Name of file to dedisperse
		self.root_directory 				=		Path(os.path.dirname(self.filename))		#Base directory where data is saved
		self.output_directory				=		Path(output_directory)		#Base directory where data should be saved in (output directory)
		self.file_type 						= 		str(self.filename).split(".")[-1]
		#self.list_of_files 				= 		self.root_filename#list(self.root_filename.glob(f"*.{self.file_type}"))
		self.header 						=		utils._open_header(str(self.filename),str(self.file_type))
		self.subbanded						=		subbanded #Can be False if no subbanded is wanted
		self._initial_downsampling_factor 	= 		1
		self.number_of_subint				=		nsubint
		self.no_zdot						=		no_zdot
		self.no_rfi_cleaning 				= 		no_rfi_cleaning
		self.use_astro_accelerate			=		use_astro_accelerate
		self.ddplan_instance 				= 		ddplan(config,'empty',self.subbanded, str(self.filename))
		self.original_dynamic_spectrum = np.zeros((self.header.nchans,1))
		self._dm_time_array = np.zeros((1,1))
		self.subint_indeces = np.zeros((1))
		self.number_of_subint_array = np.zeros((1))
		self._group_time = 0
		self._loading_data_time  = 0
		self._downsampling_time = 0
		self._dedispersing_time = 0
		self._saving_time = 0
			
	@property
	def dm_time_array(self):
		return (self._dm_time_array)
	
	@property
	def initial_downsampling_factor(self):
		return (self._initial_downsampling_factor)
	
	def create_ddplan(self):
		print("Calculating ddplan for the given configuration")
		self.ddplan_instance.calculate_ddplan()
		self.ddplan_instance.calculate_new_indeces()
		self.ddplan_instance.print_ddplan()
		self._downsample_factors = (np.ones((len(self.ddplan_instance.old_ddplan_dm_min)))*self.config.width_step).astype(int)
		self._downsample_factors[0] = 1
		print('Sampling time was ', self.header.tsamp, 'but it will be ', self.ddplan_instance.config.tsamp)
		self.config.tsamp = self.ddplan_instance.config.tsamp
		self._initial_downsampling_factor = int(self.config.tsamp/self.header.tsamp)
	
	def normalise_data(self,data):
		"""
		Normalise channels of the dynamic spectrum and return copy of clean data
		"""
		m = np.median(data,axis=1,keepdims=True)
		s = np.std(data,axis=1,keepdims=True)
		s[s==0]=1
		data -= m
		data /= s
		return data
	
	def calculate_subint_indeces(self):
		"""
		Function that calculates the subintegration indeces to append together to perform 
		dedispersion to PSRFITS file.
		"""
		#Calculate maximum time delay possible given by observing configuration and max DM.
		max_dispersion = self.calculate_max_dispersion(self.header.ftop,self.header.fbottom,self.config.dm_max)
		#The maximum time delay is used to overlap subintegrations.
		self.subint_indeces = np.arange(0,self.header.nsubint,int(self.number_of_subint-max_dispersion/self.header.tsubint))
		#The indices would go from 0 until the nsubint in steps of number_of_subint, applying the overlap.
		self.number_of_subint_array = np.ones((len(self.subint_indeces)),dtype=int)*self.number_of_subint
		self.number_of_subint_array[-1] = int(self.header.nsubint - self.subint_indeces[-1])
		
	def calculate_indeces(self):
		"""
		Function that calculates the subintegration indeces to append together to perform 
		dedispersion to filterbank files.
		"""
		#Calculate maximum time delay possible given by observing configuration and max DM.
		max_dispersion = self.calculate_max_dispersion(self.header.ftop,self.header.fbottom,self.config.dm_max)
		#The maximum time delay is used to overlap blocks.
		self.file_indeces = np.arange(0,self.header.nsamples,int((self.number_of_subint-max_dispersion)/self.header.tsamp))
		self.number_of_samples_array = np.ones((len(self.file_indeces)),dtype=int)*int(self.number_of_subint/self.header.tsamp)
		self.number_of_samples_array[-1] = int(self.header.nsamples - self.file_indeces[-1])
		#ASSUME BLOCKS OF 30s AND TAKE INTO ACCOUNT INITIAL DOWNSAMPLING i.e. smth like self.indeces = np.arange(0,myFil.header.nsamples//initial_downsampling,closest_int_to_1024(30/(tsamp*initial_downsampling)))
		#ALSO TAKE CARE OF EDGE CASES
		
		
		
	def dedisperse_psr_fits(self,filename):
		self.calculate_subint_indeces()
		start_time = time.time()
		with h5py.File(filename.replace(str(self.root_directory),str(self.output_directory)).replace("sf","hdf5"),"w") as g:
			for i in range(len(self.subint_indeces)):
				group_time_0 = time.time()
				group = g.create_group(str(i))
				group.attrs["files_dedispersed"] = filename.replace(str(self.root_directory),"")
				group.attrs["total_length"] = self.number_of_subint*self.header.tsubint
				group.attrs["length_per_file"] = self.header.tobs
				print("Importing dynamic spectrum")
				group_time_1 = time.time()
				self._group_time += group_time_1 - group_time_0
				load_time_0 = time.time()
				self.original_dynamic_spectrum = utils._load_data(filename, self.file_type, self.subint_indeces[i], self.number_of_subint_array[i], self.header, self.no_rfi_cleaning, self.no_zdot)
				load_time_1 = time.time()
				self._loading_data_time += load_time_1 - load_time_0
				for dm_step_counter, dm_step in enumerate(self.ddplan_instance.old_ddplan_dm_step):
					#Find how many dm bins you require for the given DM chunk in the ddplan
					if dm_step_counter < 1:
						print("Creating DM-time array for DMs ",0, self.ddplan_instance.dm_boundaries[dm_step_counter])
						dm_bins = int(np.ceil(self.ddplan_instance.dm_boundaries[dm_step_counter]/dm_step)) + 1
					else:
						dm_bins = int(np.ceil(self.ddplan_instance.dm_boundaries[dm_step_counter]/dm_step)) - int(self.ddplan_instance.dm_boundaries[dm_step_counter-1]/dm_step) +1
						print("Creating DM-time array for DMs ", self.ddplan_instance.dm_boundaries[dm_step_counter-1], self.ddplan_instance.dm_boundaries[dm_step_counter])
					self._dm_time_array = np.zeros((dm_bins,1))
					if self.subbanded:
						self.subband_data()
					print("Downsampling the filterbank file.")
					down_time_0 = time.time()
					self.dynamic_spectrum = block_reduce(self.original_dynamic_spectrum, block_size=(1,self.ddplan_instance.old_ddplan_downsampling_factor[dm_step_counter].astype(int)*self.initial_downsampling_factor),func=np.mean)
					down_time_1 = time.time()
					self._downsampling_time += down_time_1 - down_time_0
					print("Dedispersing the file.")
					dedisp_time_0 = time.time()
					if dm_step_counter < 1:
						dm_time_array = utils.transform(self.dynamic_spectrum, self.header.ftop,self.header.fbottom,self.ddplan_instance.old_ddplan_downsampling_factor[dm_step_counter].astype(int)*self.initial_downsampling_factor*self.header.tsamp, 0, self.ddplan_instance.dm_boundaries[dm_step_counter], self.ddplan_instance.old_ddplan_dm_step[dm_step_counter]).data
						group.attrs["dm_range_"+str(dm_step_counter)] = [0,self.ddplan_instance.dm_boundaries[dm_step_counter]]
					else:
						dm_time_array = utils.transform(self.dynamic_spectrum, self.header.ftop,self.header.fbottom,self.ddplan_instance.old_ddplan_downsampling_factor[dm_step_counter].astype(int)*self.initial_downsampling_factor*self.header.tsamp, self.ddplan_instance.dm_boundaries[dm_step_counter-1], self.ddplan_instance.dm_boundaries[dm_step_counter], self.ddplan_instance.old_ddplan_dm_step[dm_step_counter]).data
						group.attrs["dm_range_"+str(dm_step_counter)] = [self.ddplan_instance.dm_boundaries[dm_step_counter-1],self.ddplan_instance.dm_boundaries[dm_step_counter]]
					dedisp_time_1 = time.time()
					self._dedispersing_time += dedisp_time_1 - dedisp_time_0
					self._dm_time_array = np.hstack((self.dm_time_array,dm_time_array))
					self._dm_time_array = self.dm_time_array[::,1::]
					group.attrs["dm_step_"+str(dm_step_counter)] = self.ddplan_instance.old_ddplan_dm_step[dm_step_counter]
					dataset_name = str(dm_step_counter)
					save_0 = time.time()
					group.create_dataset(dataset_name,data=self.dm_time_array)
					save_1 = time.time()
					self._saving_time += save_1 - save_0
		
		end_time = time.time()
		print('That took ', round(end_time-start_time,4), ' s.')

	def dedisperse_fil(self,filename):
		self.calculate_indeces()
		with h5py.File(filename.replace(str(self.root_directory),str(self.output_directory)).replace("fil","hdf5"),"w") as g:
			for i in range(len(self.file_indeces)):
				group_time_0 = time.time()
				group = g.create_group(str(i))
				group.attrs["files_dedispersed"] = filename.replace(str(self.root_directory),"")
				group.attrs["total_length"] = self.number_of_subint
				group.attrs["length_per_file"] = self.header.tobs
				self.original_dynamic_spectrum = utils._load_data(filename, self.file_type, self.file_indeces[i], self.number_of_samples_array[i], self.header, self.no_rfi_cleaning ,self.no_zdot)
				for dm_step_counter, dm_step in enumerate(self.ddplan_instance.old_ddplan_dm_step):
					#Find how many dm bins you require for the given DM chunk in the ddplan
					if dm_step_counter < 1:
						print("Creating DM-time array for DMs ",0, self.ddplan_instance.dm_boundaries[dm_step_counter])
						dm_bins = int(np.ceil(self.ddplan_instance.dm_boundaries[dm_step_counter]/dm_step)) + 1
					else:
						dm_bins = int(np.ceil(self.ddplan_instance.dm_boundaries[dm_step_counter]/dm_step)) - int(self.ddplan_instance.dm_boundaries[dm_step_counter-1]/dm_step) +1
						print("Creating DM-time array for DMs ", self.ddplan_instance.dm_boundaries[dm_step_counter-1], self.ddplan_instance.dm_boundaries[dm_step_counter])
					self._dm_time_array = np.zeros((dm_bins,1))
					if self.subbanded:
						self.subband_data()
					print("Downsampling the filterbank file.")
					self.dynamic_spectrum = block_reduce(self.original_dynamic_spectrum, block_size=(1,self.ddplan_instance.old_ddplan_downsampling_factor[dm_step_counter].astype(int)*self.initial_downsampling_factor),func=np.mean)
					print("Dedispersing the file.")
					if dm_step_counter < 1:
						dm_time_array = utils.transform(self.dynamic_spectrum, self.header.ftop,self.header.fbottom,self.ddplan_instance.old_ddplan_downsampling_factor[dm_step_counter].astype(int)*self.initial_downsampling_factor*self.header.tsamp, 0, self.ddplan_instance.dm_boundaries[dm_step_counter], self.ddplan_instance.old_ddplan_dm_step[dm_step_counter]).data
						group.attrs["dm_range_"+str(dm_step_counter)] = [0,self.ddplan_instance.dm_boundaries[dm_step_counter]]
					else:
						dm_time_array = utils.transform(self.dynamic_spectrum, self.header.ftop,self.header.fbottom,self.ddplan_instance.old_ddplan_downsampling_factor[dm_step_counter].astype(int)*self.initial_downsampling_factor*self.header.tsamp, self.ddplan_instance.dm_boundaries[dm_step_counter-1], self.ddplan_instance.dm_boundaries[dm_step_counter], self.ddplan_instance.old_ddplan_dm_step[dm_step_counter]).data
						group.attrs["dm_range_"+str(dm_step_counter)] = [self.ddplan_instance.dm_boundaries[dm_step_counter-1],self.ddplan_instance.dm_boundaries[dm_step_counter]]
					self._dm_time_array = np.hstack((self.dm_time_array,dm_time_array))
					self._dm_time_array = self.dm_time_array[::,1::]
					group.attrs["dm_step_"+str(dm_step_counter)] = self.ddplan_instance.old_ddplan_dm_step[dm_step_counter]
					dataset_name = str(dm_step_counter)
					group.create_dataset(dataset_name,data=self.dm_time_array)


	def aa_dedisperse_fil(self,filename):
		self.calculate_indeces()
		with h5py.File(filename.replace(str(self.root_directory),str(self.output_directory)).replace("fil","hdf5"),"w") as g:
			for i in range(len(self.file_indeces)):
				group_time_0 = time.time()
				group = g.create_group(str(i))
				group.attrs["files_dedispersed"] = filename.replace(str(self.root_directory),"")
				group.attrs["total_length"] = self.number_of_subint
				group.attrs["length_per_file"] = self.header.tobs
				
				sigproc_input = aa_py_sigproc_input(filename)
				metadata = sigproc_input.read_metadata()
				if not sigproc_input.read_signal():
					print("ERROR: Invalid .fil file path. Exiting...")
					sys.exit()
				input_buffer = sigproc_input.input_buffer()

				# ddtr_plan settings
				# settings: aa_py_dm(low, high, step, inBin, outBin)
				temp_list = []
				for i in range(len(self.ddplan_instance.old_ddplan_dm_step)):
					tmp=None
					if i<1:
						lowDM = 0
					else:
						lowDM = self.ddplan_instance.dm_boundaries[i-1]
					highDM = self.ddplan_instance.dm_boundaries[i]
					tmp = aa_py_dm(lowDM,highDM,self.ddplan_instance.old_ddplan_dm_step[i],1,self.ddplan_instance.old_ddplan_downsampling_factor[i].astype(int)*self.initial_downsampling_factor)
					temp_list.append(tmp)
				dm_list = np.array(temp_list,dtype=aa_py_dm)
				# Create ddtr_plan
				self.ddtr_plan = aa_py_ddtr_plan(dm_list)
				enable_msd_baseline_noise=False
				self.ddtr_plan.set_enable_msd_baseline_noise(enable_msd_baseline_noise)
				self.ddtr_plan.print_info()
				# Set up pipeline components
				pipeline_components = aa_py_pipeline_components()
				pipeline_components.dedispersion = True
				# Set up pipeline component options	
				pipeline_options = aa_py_pipeline_component_options()
				pipeline_options.output_dmt = True
				#Need to be enabled otherwise there will be no data copy from GPU memory to host memory
				pipeline_options.copy_ddtr_data_to_host = True
				# Select GPU card number on this machine
				card_number = 0
				# Create pipeline
				pipeline = aa_py_pipeline(pipeline_components, pipeline_options, metadata, input_buffer, card_number)
				pipeline.bind_ddtr_plan(self.ddtr_plan) # Bind the ddtr_plan
				# Run the pipeline with AstroAccelerate
				while (pipeline.run()):
					print("NOTICE: Python script running over next chunk")
					if pipeline.status_code() == -1:
						print("ERROR: Pipeline status code is {}. The pipeline encountered an error and cannot continue.".format(pipeline.status_code()))
						break
				# Get the data of DDTR to python
				(ts_inc, ddtr_output) = pipeline.get_buffer()
				print("Dedispersion finished, now saving data")

				for dm_step_counter in range(pipeline.ddtr_range()):
					list_ndms = pipeline.ddtr_ndms()
					nsamps = int(ts_inc)
					if dm_step_counter < 1:
						group.attrs["dm_range_"+str(dm_step_counter)] = [0,self.ddplan_instance.dm_boundaries[dm_step_counter]]
					else:
						group.attrs["dm_range_"+str(dm_step_counter)] = [self.ddplan_instance.dm_boundaries[dm_step_counter-1],self.ddplan_instance.dm_boundaries[dm_step_counter]]
					
					dm_time_array = np.zeros((list_ndms[dm_step_counter],nsamps))
					for idm in range(list_ndms[dm_step_counter]):
						dm_time_array[idm] = np.ctypeslib.as_array( ddtr_output[dm_step_counter][idm] , (int(nsamps),))

					# TODO: Downsampling is done here, ideally it should be done by astro-accelerate
					dm_time_array = block_reduce(dm_time_array, block_size=(1,self.ddplan_instance.old_ddplan_downsampling_factor[dm_step_counter].astype(int)*self.initial_downsampling_factor),func=np.mean)
					group.attrs["dm_step_"+str(dm_step_counter)] = self.ddplan_instance.old_ddplan_dm_step[dm_step_counter]
					dataset_name = str(dm_step_counter)
					group.create_dataset(dataset_name,data=dm_time_array)


	def subband_data(self):
		chantop = int((self.header.ftop - self.subbanded[0])/np.abs(self.header.foff))
		chanbottom = int((self.header.ftop - self.subbanded[1])/np.abs(self.header.foff))
		self.original_dynamic_spectrum = self.original_dynamic_spectrum[chantop:chanbottom]
		self.header.subband_data(self.header.ftop-chantop*np.abs(self.header.foff),self.header.ftop-chanbottom*np.abs(self.header.foff))
	
	def calculate_max_dispersion(self,ftop,fbottom,DM):
		return self.config.KDM * DM * (fbottom**-2-ftop**-2)
		
	def dedisperse_multiple_files(self):
		for file in self.list_of_files:
			self.dedisperse_psr_fits(file)
			
	def dedisperse(self,index):
		#self.create_ddplan()
		if self.file_type == 'sf':
			self.dedisperse_psr_fits(str(self.filename))
		elif self.file_type == 'fil' and self.use_astro_accelerate:
			self.aa_dedisperse_fil(str(self.filename))
		else:
			self.calculate_indeces()
			self.dedisperse_fil(index)
			return self.return_dm_time

def parse_args():
	parser = argparse.ArgumentParser(
		description="Create DM-time images for Mask RCNN inference and save them to hdf5 file.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument('-aa','--use-astro-accelerate', dest='use_astro_accelerate', help='Use AstroAccelerate for dedispersion', action='store_true')
	parser.add_argument('--no-zdot', dest='nozdot', help='Disable Z-Dot filter', action='store_true')
	parser.add_argument('--no-rfi-cleaning', dest='norficleaning', help='Disable IQRM RFI cleaning', action='store_true')
	parser.add_argument('-o', '--outdir', help='Output directory for saving hdf5 file', default=os.getcwd())
	parser.add_argument('-i','--datapath', help = 'Filename of the filterbank file to dedisperse', default=os.getcwd())
	parser.add_argument('--nchunk', type=int, help='Number of chunks the search will be split into (Number of sub-integrations for PSRF FITS file or seconds of data for filterbank file)',default=100)
	parser.add_argument('--subband',help='Top frequency and bottom frequency channel to subband the data to. This should be a comma-separated list of the values in MHz.', default=False)
	return parser.parse_args()


if __name__ == '__main__':
	args = parse_args()
	config = Config()
	start_time = time.time()
	dedisperse_class = Dedispersing_files(config, args.datapath, args.outdir, args.subband, args.nchunk, args.norficleaning, args.nozdot, args.use_astro_accelerate)
	dedisperse_class.dedisperse()
	end_time = time.time()
	print('That took ', round(end_time-start_time,4), ' s.')
	

