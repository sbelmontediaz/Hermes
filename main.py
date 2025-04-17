import numpy as np
import argparse
import sys, os
from config import Config

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import csv
import glob
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.models.segmentation
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.colors as mcolors
import utils
from pathlib import Path
from tqdm import tqdm
import time
import functools as ft
from skimage.measure import block_reduce
from astropy.stats import sigma_clipped_stats

import logging
import math
import glob
from ddplan import ddplan
from scipy.signal import savgol_filter
from astropy.stats import sigma_clipped_stats
import numbers
from numpy.lib.stride_tricks import as_strided

from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from collections import deque

from queue import Queue, Empty, Full
from concurrent.futures import ThreadPoolExecutor
import threading

#List of colours to produce the plots.
color_list = np.array(['tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan','tab:blue'])
#Headers of the information saved in the csv file.
column_names = ['ID','DM_time_range','width_(\u00B5s)', 'time_(s)', 'DM', 'Prediction_score' ,'Burst in image', 'slice_num','rows_in_slice','cols_in_slice','filterbank_file','time_inside_filterbank','ml_prediction_image_name']



#Class to perform data processing of the DM-time transform. It performs the downsampling, slicing and normalisation.
class Data_processing(object):

	def __init__(self, config, dm_time_array):
		self.config = config
		self.dm_time_array = dm_time_array
		self.slices_dm_time_array = np.zeros((1,1,1,1))
		self.slices_num_of_rows = 0
		self.slices_num_of_cols = 0
		self.mean_set = 0
		self.std_set = 0
		self._loading_time = 0
		self._downsampling_time = 0
		self._slicing_time = 0
		self._normalising_time = 0

	#Function that uses strides to change the way the data is accessed to efficiently slice a 2d array. The slicing is performed with an overlap between slices.
	def _slice_2d_array_into_slices(self,array, slice_size, overlap):
		num_rows, num_cols = array.shape
		step_size = slice_size - overlap
		num_slices_row = (num_rows - slice_size) // step_size + 1
		num_slices_col = (num_cols - slice_size) // step_size + 1
		shape = (num_slices_row, num_slices_col, slice_size, slice_size)
		strides = (array.strides[0] * step_size, array.strides[1] * step_size, array.strides[0], array.strides[1])
		slices = np.lib.stride_tricks.as_strided(array, shape = shape, strides = strides)
		return slices

	#Calculate the most optimal time window size based on the chosen S/N drop factor. The optimal time window is calculated to be an integer number of the window size.
	def calculate_time_window(self,width,downsampling_fact):
		optimal_time_window = 2*(self.config.fudge_factor*width*10**-6)/(self.config.tsamp*downsampling_fact)

		if round(((optimal_time_window)%self.config.window_size)/self.config.window_size) == 1:
			time_window_size = optimal_time_window + self.config.window_size-optimal_time_window%self.config.window_size
	  
		else:
			time_window_size = optimal_time_window - optimal_time_window%self.config.window_size

		return time_window_size

	#Function to downsample the current DM-time array and merge with the next DM range.
	def downsample_and_merge(self,group,index):
		print("Downsampling and merging the DM-time array.")
		start_time = time.time()
		self.downsample_dm_time_array(2,2)
		downsampled_previous = self.dm_time_array.copy()
		end_time = time.time()
		print("That took ", round(end_time-start_time,4), " s.")
		self._downsampling_time += end_time - start_time
		self.obtain_dm_time_array(group,index)
		#Check if the time axis matches between the two arrays (sometimes downsampling by two can make the array have an extra time sample).
		if downsampled_previous.shape[1] != self.dm_time_array.shape[1]:
			size1 = downsampled_previous.shape[1]
			size2 = self.dm_time_array.shape[1]
			difference_in_shape = np.abs(downsampled_previous.shape[1]-self.dm_time_array.shape[1])
			if size1>size2:
				downsampled_previous = downsampled_previous[::,:-difference_in_shape]
			else:
				self.dm_time_array = self.dm_time_array[::,:-difference_in_shape]

		self.dm_time_array = np.vstack((downsampled_previous,self.dm_time_array))


	#Downsample a 2d array by using strides and summing for more efficiency.
	def downsample_dm_time_array(self, index, dm_factor, time_factor):
		new_shape = (self.dm_time_array[index].shape[0]//dm_factor, dm_factor, self.dm_time_array[index].shape[1]//time_factor, time_factor)
		strides = (self.dm_time_array[index].strides[0] * dm_factor, self.dm_time_array[index].strides[0], self.dm_time_array[index].strides[1] * time_factor, self.dm_time_array[index].strides[1])
		self.dm_time_array[index] = np.lib.stride_tricks.as_strided(self.dm_time_array[index], shape = new_shape, strides = strides)
		self.dm_time_array[index] = self.dm_time_array[index].sum(axis=(1,3))/(time_factor*dm_factor)	
		
	#Function to slice the array and normalise the slices by scaling the set to have zero mean and std of unity.
	def slice_and_normalize(self,index):
		print("Slicing DM-time array number ", index)
		#If index is larger than max DM range, do not merge, just downsample
		if index >= self.dm_time_array.shape[0]:
			index = self.dm_time_array.shape[0]-1
			self.downsample_dm_time_array(index,2,2)
		elif index != 0:
			self.downsample_dm_time_array(index-1,2,2)
			if self.dm_time_array[index-1].shape[1] != self.dm_time_array[index].shape[1]:
				size1 = self.dm_time_array[index-1].shape[1]
				size2 = self.dm_time_array[index].shape[1]
				difference_in_shape = np.abs(size1-size2)
				if size1>size2:
					self.dm_time_array[index-1] = self.dm_time_array[index-1][::,:-difference_in_shape]
				else:
					self.dm_time_array[index] = self.dm_time_array[index][::,:-difference_in_shape]

			self.dm_time_array[index] = np.vstack((self.dm_time_array[index-1],self.dm_time_array[index]))
		self.slices_dm_time_array = self._slice_2d_array_into_slices(self.dm_time_array[index].copy(),self.config.image_size,self.config.overlap)
		self.slices_num_of_rows, self.slices_num_of_cols = self.slices_dm_time_array.shape[0:2]
		self.mean_set = np.mean(self.slices_dm_time_array)
		self.std_set = np.std(self.slices_dm_time_array)
		self.slices_dm_time_array -= self.mean_set
		self.slices_dm_time_array /= self.std_set
		self.slices_dm_time_array = self.slices_dm_time_array.reshape(self.slices_num_of_rows*self.slices_num_of_cols,self.config.image_size,self.config.image_size)

		
#Class to create a dataset to later create a dataloader.
class MyDataset(Dataset):

	def __init__(self, data,transform=None):
		self.data = data
		self.transform = transform
	
	def __len__(self):
		return self.data.shape[0]
		
	def __getitem__(self, index):
		image = self.data[index]
		label = 0
		if self.transform:
			image = self.transform(image)
		return image, label

#Class that incorporates the change in dynamic range in each image as a transformation for the dataloader.
class Normalize_DM_time_snap(object):
	#The pixel values are re-scaled so that 12 stds fit in the range 0-255, and the mean is located so that 4 stds fit in the low range and 8 in the high range.
	def __call__(self,image):
		image *= 255/12
		image += 255*4/12
		return image
		
#Class to transform the array into a tensor of shape [3,x,y]
class ToTensor(object):
	def __call__(self,image):
		image.shape = [1,image.shape[0],image.shape[1]]
		image = np.vstack((image,image,image))
		image = torch.as_tensor(image, dtype=torch.float32)
		return image

class Hermes(object):
	def __init__(self,config, filename, output_directory, subbanded, transforms, nsubint=100, no_rfi_cleaning=False, no_zdot=False):
		self.config 						=		config
		self.filename 						=		Path(filename)				#Name of file to dedisperse
		self.output_directory				=		Path(output_directory)		#Base directory where data should be saved in (output directory)
		self.file_type 						= 		str(self.filename).split(".")[-1]
		self.header 						=		utils._open_header(str(self.filename),str(self.file_type))
		self.subbanded						=		subbanded #Can be False if no subbanded is wanted
		
		#From dedisperse class
		self._initial_downsampling_factor 	= 		1
		self.number_of_subint				=		nsubint
		self.no_zdot						=		no_zdot
		self.no_rfi_cleaning 				= 		no_rfi_cleaning
		self.ddplan_instance 				= 		ddplan(config,'empty',self.subbanded, str(self.filename))
		
		#From search class
		self._dm_time_array = np.empty((1,1,1))
		self.data_processing = Data_processing(self.config,self.dm_time_array)
		self.transforms = transforms
		self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
		self.model = self._initialize_model()
		#self._create_directories()
		self.predictions = []
		self.batch_index = 0
		self.group_index = 0
		self.prediction_id = 0
		self.prediction_information = []
		self.prediction_information_to_csv = np.zeros((1,13))
		self.width = self.config.min_width
		self.window_idx = int(np.log(self.width)/np.log(self.config.width_step))
		self.prediction_idx = np.zeros((1))
		self.time = 0
		self.number_of_images = 0
		self.num_imgs = []
		self.num_rows = []
		self.num_cols = []
		self.processing_time = 0
		self.plot_time = 0
		self.redownsample_factor = 1
		self.counter = 0	
		self.inference_time = 0	
		
		#Multithreading
		self.slice_queue = Queue(maxsize=20)
		self.results_queue = Queue(maxsize=20)
		self.executor = ThreadPoolExecutor(max_workers=20)
		self.plotting_thread = threading.Thread(target=self.plot_results_pipeline, daemon=True)
		self.plotting_thread.start()
		
			
	@property
	def dm_time_array(self):
		return (self._dm_time_array)
	
	@property
	def initial_downsampling_factor(self):
		return (self._initial_downsampling_factor)
	
	def create_ddplan(self):
		"""
		Initialises and calculates a dedispersion plan based on the config and observation metadata.
		Also adjusts the time sampling.
		"""
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
		
	def calculate_indeces(self):
		"""
		Calculates the time/sample block indices for dedispersion of filterbank or PSRFITS data.
		Accounts for overlap based on max dispersion delay.
		"""
		#Calculate maximum time delay possible given by observing configuration and max DM.
		max_dispersion = self.calculate_max_dispersion(self.header.ftop,self.header.fbottom,self.config.dm_max)
		if self.file_type == "sf":
			#The maximum time delay is used to overlap subintegrations.
			self.file_indeces = np.arange(0,self.header.nsubint,int(self.number_of_subint-max_dispersion/self.header.tsubint))
			#The indices would go from 0 until the nsubint in steps of number_of_subint, applying the overlap.
			self.number_of_samples_array = np.ones((len(self.subint_indeces)),dtype=int)*self.number_of_subint
			self.number_of_samples_array[-1] = int(self.header.nsubint - self.subint_indeces[-1])
		else:
			#The maximum time delay is used to overlap blocks.
			self.file_indeces = np.arange(0,self.header.nsamples,int((self.number_of_subint-max_dispersion)/self.header.tsamp))
			#The indices would go from 0 until the nsample in steps of number_of_subint (in this case seconds), applying the overlap.
			self.number_of_samples_array = np.ones((len(self.file_indeces)),dtype=int)*int(self.number_of_subint/self.header.tsamp)
			self.number_of_samples_array[-1] = int(self.header.nsamples - self.file_indeces[-1])
		

	def dedisperse(self,index):
		"""
		Runs the dedispersion process for the selected time chunk.
		Generates a DM-time image array for multiple DM steps defined in the DD plan.
		
		Parameters:
			index (int): Time index to dedisperse.
		
		Returns:
			None: Updates internal DM-time array.
		"""
		self.original_dynamic_spectrum = utils._load_data(str(self.filename), self.file_type, self.file_indeces[index], self.number_of_samples_array[index], self.header, self.no_rfi_cleaning ,self.no_zdot)
		self.return_dm_time = np.empty(len(self.ddplan_instance.old_ddplan_dm_step),dtype=object)
		for dm_step_counter, dm_step in enumerate(self.ddplan_instance.old_ddplan_dm_step):
			#Find how many dm bins you require for the given DM chunk in the ddplan
			if dm_step_counter < 1:
				print("Creating DM-time array for DMs ",0, self.ddplan_instance.dm_boundaries[dm_step_counter])
				dm_bins = int(np.ceil(self.ddplan_instance.dm_boundaries[dm_step_counter]/dm_step)) + 1
			else:
				dm_bins = int(np.ceil(self.ddplan_instance.dm_boundaries[dm_step_counter]/dm_step)) - int(self.ddplan_instance.dm_boundaries[dm_step_counter-1]/dm_step) +1
				print("Creating DM-time array for DMs ", self.ddplan_instance.dm_boundaries[dm_step_counter-1], self.ddplan_instance.dm_boundaries[dm_step_counter])
			#self._dm_time_array = np.zeros((dm_bins,1))
			if self.subbanded:
				self.subband_data()
			print("Downsampling the filterbank file.")
			self.dynamic_spectrum = block_reduce(self.original_dynamic_spectrum, block_size=(1,self.ddplan_instance.old_ddplan_downsampling_factor[dm_step_counter].astype(int)*self.initial_downsampling_factor),func=np.mean)
			print("Dedispersing the file.")
			if dm_step_counter < 1:
				self._dm_time_array = utils.transform(self.dynamic_spectrum, self.header.ftop,self.header.fbottom,self.ddplan_instance.old_ddplan_downsampling_factor[dm_step_counter].astype(int)*self.initial_downsampling_factor*self.header.tsamp, 0, self.ddplan_instance.dm_boundaries[dm_step_counter], self.ddplan_instance.old_ddplan_dm_step[dm_step_counter]).data
			else:
				self._dm_time_array = utils.transform(self.dynamic_spectrum, self.header.ftop,self.header.fbottom,self.ddplan_instance.old_ddplan_downsampling_factor[dm_step_counter].astype(int)*self.initial_downsampling_factor*self.header.tsamp, self.ddplan_instance.dm_boundaries[dm_step_counter-1], self.ddplan_instance.dm_boundaries[dm_step_counter], self.ddplan_instance.old_ddplan_dm_step[dm_step_counter]).data
			#self._dm_time_array = np.hstack((self.dm_time_array,dm_time_array))
			#self._dm_time_array = self.dm_time_array[::,1::]
			self.return_dm_time[dm_step_counter] = self.dm_time_array
					
					
	def subband_data(self):
		chantop = int((self.header.ftop - self.subbanded[0])/np.abs(self.header.foff))
		chanbottom = int((self.header.ftop - self.subbanded[1])/np.abs(self.header.foff))
		self.original_dynamic_spectrum = self.original_dynamic_spectrum[chantop:chanbottom]
		self.header.subband_data(self.header.ftop-chantop*np.abs(self.header.foff),self.header.ftop-chanbottom*np.abs(self.header.foff))
	
	def calculate_max_dispersion(self,ftop,fbottom,DM):
		"""
		Calculates the maximum dispersion delay (in seconds) between two frequencies for a given DM.
		
		Returns:
			float: Time delay in seconds
		"""
		return self.config.KDM * DM * (fbottom**-2-ftop**-2)
			
			
	#Load the Mask RCNN model from the torchvision library to the GPU and set it to inference mode.
	def _initialize_model(self):
		"""
		Loads and initializes the Mask R-CNN model with pretrained weights.
		
		Returns:
			model (torch.nn.Module): The loaded and configured Mask R-CNN model.
		"""
		torch.hub.set_dir(self.config.resnet_weights_directory)
		model=torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, min_size=self.config.image_size,max_size=self.config.image_size)
		in_features = model.roi_heads.box_predictor.cls_score.in_features
		model.roi_heads.box_predictor=FastRCNNPredictor(in_features,num_classes=self.config.num_of_classes)
		model.load_state_dict(torch.load(self.config.weights_directory))
		model.to(self.device)
		model.eval()
		return model
		
	#Create output directories
	def _create_directories(self,group_number):
		"""
		Creates output directories for saving search results, one per data chunk.
		
		Parameters:
			group_number (int): Number of subdirectories to create.
		"""
		for number in range(group_number):
			if not os.path.exists(str(self.output_directory / Path(str(number)))):
				os.makedirs(str(self.output_directory / Path(str(number))))
		
			
	#Function to run inference on the Mask RCNN model and collect the results.
	
	def inference(self, dataset):
		"""
		Runs inference on the Mask R-CNN model for all DM-time image slices.
		
		Parameters:
			dataset (DataLoader): The dataloader containing the preprocessed DM-time image slices.
		
		Collects:
			Predictions (with scores above threshold) and relevant metadata for later plotting and analysis.
		"""
		self.batch_index = 0
		self.collected_predictions = deque()  # Collect predictions + metadata

		with torch.no_grad():
			for i, (image_batch, _) in enumerate(tqdm(dataset, desc="Running inference.", leave=True)):
				image_batch = image_batch.to(self.device)
				start_time = time.time()
				batch_predictions = self.model(image_batch)
				end_time = time.time()
				self.inference_time += end_time - start_time

				for j, prediction in enumerate(batch_predictions):
					if len(prediction['scores']) == 0 or prediction['scores'][0].item() < self.config.threshold:
						continue

					pred = {
						k: v[:9].cpu().detach() if torch.is_tensor(v) else v
						for k, v in prediction.items()
					}
					image = image_batch[j].cpu()
					slice_idx = self.batch_index + j

					self.collected_predictions.append({
						"prediction": pred,
						"image": image,
						"slice_idx": slice_idx,
						"dm_index": self.dm_index,
						"time_index": self.time_index,
						"file_offset": self.file_indeces[self.time_index] * self.header.tsamp
					})

				self.batch_index += len(image_batch)
	
					
	#Main function to peform the search.
	def Mask_RCNN_inference(self, index):
		"""
		Main function to run a Mask R-CNN-based search for a specific width index.
		
		Parameters:
			index (int): DM index of the DM-time plane to process for a specific burst width.
		"""
		print("Searching for bursts in index ", index)
		self.width = self.config.width_array[index]
		print("Searching for bursts of width ", self.width, " \u00B5s.")
		#Obtain a DM-time array, slice it and load it to the dataloader.
		self.data_processing.slice_and_normalize(index)
		self.number_of_images += self.data_processing.slices_dm_time_array.shape[0]
		self.num_imgs.append(self.data_processing.slices_dm_time_array.shape[0])
		self.num_rows.append(self.data_processing.slices_num_of_rows)
		self.num_cols.append(self.data_processing.slices_num_of_cols)
		dataset = DataLoader(MyDataset(self.data_processing.slices_dm_time_array, self.transforms),batch_size=self.config.batch_size,shuffle=False,num_workers=8, pin_memory=True, prefetch_factor=256)
		#Run inference and collect results.		
		self.inference(dataset)
		self.plot_collected_predictions()
		print("Done. Number of images processed:", self.number_of_images)
	
	#Save the information of the candidates in a csv file to inspect later.
	def save_prediction_information(self):
		"""
		Saves all prediction metadata collected during a search iteration to a CSV file.
		This includes time, DM, bounding box info, scores, and image paths.
		"""
		data_dict_list = [dict(zip(column_names,row)) for row in self.prediction_information_to_csv[1::]]
		#with open(str(self.output_directory / Path(str(self.time_index)))+"/table_detections.csv", mode='w', newline ='') as file:
		with open(str(self.output_directory)+"/table_detections.csv", mode='w', newline ='') as file:
			writer = csv.DictWriter(file, fieldnames = column_names)
			writer.writeheader()
			writer.writerows(data_dict_list)
			
	#Function to plot the burst candidates and save key information.
	
	def plot_collected_predictions(self):
		"""
		Plots and saves all predictions collected during inference.
		
		Each prediction is drawn over the original DM-time image, with bounding boxes and labels.
		Time and DM axes are scaled, and the output is saved as a PNG file.
		"""
		for item in self.collected_predictions:
			prediction = item["prediction"]
			image = item["image"]
			slice_idx = item["slice_idx"]
			dm_index = item["dm_index"]
			time_index = item["time_index"]
			file_offset = item["file_offset"]

			binary_mask = np.zeros((self.config.image_size, self.config.image_size))
			image_np = image.numpy()
			image_np = image_np[0]

			plt.figure(figsize=(8, 8))
			plt.imshow(image_np, origin="lower", aspect="auto", vmin=0, vmax=255)

			for k, score in enumerate(prediction['scores']):
				if score < self.config.threshold:
					continue

				mask = prediction['masks'][k, 0].numpy()
				box = prediction['boxes'][k].numpy()
				color = mcolors.TABLEAU_COLORS[color_list[k]]

				binary_mask += (mask > 0.5).astype(np.uint8)

				DM, time = np.where((image_np - np.min(image_np)) * (mask > 0.5) == np.max((image_np - np.min(image_np)) * (mask > 0.5)))
				DM = DM[0] if len(DM) else 0
				time = time[0] if len(time) else 0
				

				row_idx = slice_idx // self.data_processing.slices_num_of_cols
				col_idx = slice_idx % self.data_processing.slices_num_of_cols

				dm_val = ((DM + (self.config.image_size - self.config.overlap) * row_idx) *
					      self.ddplan_instance.old_ddplan_dm_step[dm_index])
				time_val = ((self.config.image_size - self.config.overlap) * col_idx + time) * \
					       self.config.tsamp * self.initial_downsampling_factor * 2**dm_index

				filename = str(self.output_directory / Path(str(time_index))) + f"/{self.counter}_{self.width}_{round(dm_val, 2)}_{round(time_val, 3)}_{slice_idx}.png"

				self.prediction_information_to_csv = np.vstack((self.prediction_information_to_csv, [
					self.prediction_id, dm_index, self.width, time_val, dm_val,
					score.item(), len(prediction['scores']), slice_idx,
					self.data_processing.slices_num_of_rows, self.data_processing.slices_num_of_cols,
					str(self.filename), time_val + file_offset, filename
				]))
				self.prediction_id += 1

				plt.hlines(box[1], box[0], box[2], colors=color, linestyles='dashed', label=f"{score:.4f}")
				plt.hlines(box[3], box[0], box[2], colors=color, linestyles='dashed')
				plt.vlines(box[0], box[1], box[3], colors=color, linestyles='dashed')
				plt.vlines(box[2], box[1], box[3], colors=color, linestyles='dashed')
				
			lower_DM = ((self.config.image_size - self.config.overlap) * row_idx) * self.ddplan_instance.old_ddplan_dm_step[dm_index]
			upper_DM = ((self.config.image_size - self.config.overlap) * (row_idx + 1)) * self.ddplan_instance.old_ddplan_dm_step[dm_index]
			
			lower_time = -(self.config.tsamp * self.initial_downsampling_factor * self.config.image_size * 2**dm_index) / 2
			upper_time = -lower_time

			plt.xticks([0,self.config.image_size//2,self.config.image_size],[round(lower_time,2),0,round(upper_time,2)],fontsize=15)
			plt.yticks([0,self.config.image_size//2,self.config.image_size],[round(lower_DM,2),round(lower_DM + (upper_DM-lower_DM)/2,2),round(upper_DM,2)],fontsize=15)
			plt.xlabel("Time range (s)", fontsize=15)
			plt.ylabel("DM (pc cm$^{-3}$)", fontsize=15)
			plt.legend(fontsize=15)

			plt.tight_layout()
			
			plt.savefig(filename,dpi=200)

			plt.close()
			self.counter += 1
			
	def _plot_prediction_item(self, item):
		"""
		Plots and saves all predictions collected during inference.
		
		Each prediction is drawn over the original DM-time image, with bounding boxes and labels.
		Time and DM axes are scaled, and the output is saved as a PNG file.
		"""
		print("LOLOL THIS IS BEING CALLED")
		prediction = item["prediction"]
		image = item["image"]
		slice_idx = item["slice_idx"]
		dm_index = item["dm_index"]
		time_index = item["time_index"]
		file_offset = item["file_offset"]
		rows = item["rows"]
		cols = item["cols"]

		binary_mask = np.zeros((self.config.image_size, self.config.image_size))
		image_np = image.numpy()
		image_np = image_np[0]

		plt.figure(figsize=(8, 8))
		plt.imshow(image_np, origin="lower", aspect="auto", vmin=0, vmax=255)

		for k, score in enumerate(prediction['scores']):
			if score < self.config.threshold:
				continue

			mask = prediction['masks'][k, 0].numpy()
			box = prediction['boxes'][k].numpy()
			color = mcolors.TABLEAU_COLORS[color_list[k]]

			binary_mask += (mask > 0.5).astype(np.uint8)

			DM, time = np.where((image_np - np.min(image_np)) * (mask > 0.5) == np.max((image_np - np.min(image_np)) * (mask > 0.5)))
			DM = DM[0] if len(DM) else 0
			time = time[0] if len(time) else 0
			
			width = self.config.width_array[time_index]
			
			row_idx = slice_idx // cols#self.data_processing.slices_num_of_cols
			col_idx = slice_idx % cols#self.data_processing.slices_num_of_cols

			dm_val = ((DM + (self.config.image_size - self.config.overlap) * row_idx) *
				      self.ddplan_instance.old_ddplan_dm_step[dm_index])
			time_val = ((self.config.image_size - self.config.overlap) * col_idx + time) * \
				       self.config.tsamp * self.initial_downsampling_factor * 2**dm_index

			#filename = str(self.output_directory / Path(str(time_index))) + f"/{self.counter}_{width}_{round(dm_val, 2)}_{round(time_val, 3)}_{slice_idx}.png"
			filename = str(self.output_directory) + f"/{self.counter}_{width}_{round(dm_val, 2)}_{round(time_val, 3)}_{slice_idx}.png"

			self.prediction_information_to_csv = np.vstack((self.prediction_information_to_csv, [
				self.prediction_id, dm_index, width, time_val, dm_val,
				score.item(), len(prediction['scores']), slice_idx,
				rows, cols,
				str(self.filename), time_val + file_offset, filename
			]))
			self.prediction_id += 1

			plt.hlines(box[1], box[0], box[2], colors=color, linestyles='dashed', label=f"{score:.4f}")
			plt.hlines(box[3], box[0], box[2], colors=color, linestyles='dashed')
			plt.vlines(box[0], box[1], box[3], colors=color, linestyles='dashed')
			plt.vlines(box[2], box[1], box[3], colors=color, linestyles='dashed')
			
		lower_DM = ((self.config.image_size - self.config.overlap) * row_idx) * self.ddplan_instance.old_ddplan_dm_step[dm_index]
		upper_DM = ((self.config.image_size - self.config.overlap) * (row_idx + 1)) * self.ddplan_instance.old_ddplan_dm_step[dm_index]
		
		lower_time = -(self.config.tsamp * self.initial_downsampling_factor * self.config.image_size * 2**dm_index) / 2
		upper_time = -lower_time

		plt.xticks([0,self.config.image_size//2,self.config.image_size],[round(lower_time,2),0,round(upper_time,2)],fontsize=15)
		plt.yticks([0,self.config.image_size//2,self.config.image_size],[round(lower_DM,2),round(lower_DM + (upper_DM-lower_DM)/2,2),round(upper_DM,2)],fontsize=15)
		plt.xlabel("Time range (s)", fontsize=15)
		plt.ylabel("DM (pc cm$^{-3}$)", fontsize=15)
		plt.legend(fontsize=15)

		plt.tight_layout()
		
		plt.savefig(filename,dpi=200)

		plt.close()
		self.counter += 1
	
	"""
	def search(self):
		'''
		Main entry point to perform a full radio transient search.
		
		Steps:
			- Create dedispersion plan
			- Dedisperse input data
			- Run Mask R-CNN inference
			- Plot results
			- Save metadata to disk
		'''
		self.create_ddplan()
		self.calculate_indeces()
		self._create_directories(len(self.file_indeces))
		for time_index in range(len(self.file_indeces)):
			#Reset all the variables to keep track of the groups, downsampling factors, etc.
			self.prediction_id = 0
			self.counter = 0
			self.prediction_information_to_csv = np.zeros((1,13))
			self.time_index = time_index
			self.dedisperse(self.time_index)
			self.data_processing = Data_processing(self.config,self.return_dm_time)
			for dm_index in range(len(self.return_dm_time)):
				self.dm_index = dm_index
				self.Mask_RCNN_inference(dm_index)
			self.save_prediction_information()
			
	"""
			
	def prepare_slices(self, time_index, dm_array):
		"""
		Dedisperse and slice a DM-time array for a given time index.
		
		Args:
			time_index (int): Index of the time chunk.
			dm_array (np.ndarray): Array returned by dedispersion, containing DM-time slices.
		"""
		processor = Data_processing(self.config,dm_array)
		for dm_index in range(len(dm_array)):
			processor.slice_and_normalize(dm_index)
			slices = processor.slices_dm_time_array.copy()
			
			self.slice_queue.put({
				"time_index": time_index,
				"dm_index": dm_index,
				"slices": slices,
				"rows": processor.slices_num_of_rows,
				"cols": processor.slices_num_of_cols
			})
	
	def run_inference_pipeline(self):
		"""
		Pulls slice batches from the queue, runs inference, and stores results for plotting.
		"""
		print("Pulling slices to GPU and running inference.")
		while not self.slice_queue.empty():
			item = self.slice_queue.get()
			slices = item["slices"]
			dm_index = item["dm_index"]
			time_index = item["time_index"]
			
			dataset = DataLoader(
				MyDataset(slices, self.transforms),
				batch_size = self.config.batch_size,
				shuffle = False,
				num_workers = 2,
				pin_memory = True,
				prefetch_factor = 128
			)
			print("Dataloader created")
			batch_index = 0
			with torch.no_grad():
				for images, _ in dataset:
					print("Running inference")
					images = images.to(self.device)
					predictions = self.model(images)
					
					for i, prediction in enumerate(predictions):
						if len(prediction['scores']) == 0 or prediction['scores'][0].item() < self.config.threshold:
							continue
						
						pred = {
							k: v[:9].cpu().detach() if torch.is_tensor(v) else v
							for k, v in prediction.items()
						}
						image = images[i].cpu()
						try:
							self.results_queue.put({
								"prediction": pred,
								"image": image,
								"slice_idx": batch_index + i,
								"dm_index": dm_index,
								"time_index": time_index,
								"file_offset": self.file_indeces[time_index] * self.header.tsamp, #modify in the future to address PSRFITS format
								"rows": item["rows"],
								"cols": item["cols"]
							}, timeout=5)
						except Full:
							print("[Warning] Results queue full - skipping this prediction to avoid stall.")
					batch_index += len(images)
				print('Finished gpu inference')
					
	def plot_results_pipeline(self):
		"""
		Continuously pulls predictions from the results queue and plots/saves images.
		"""
		while True:
			try:
				item = self.results_queue.get(timeout=1)
				if item is None:
					break
				self._plot_prediction_item(item)
			except Empty:
				time.sleep(0.1)
				continue
		"""
		while not self.results_queue.empty():
			item = self.results_queue.get()
			self._plot_prediction_item(item)
		"""
			
	def search(self):
		start_time = time.time()
		self.create_ddplan()
		self.calculate_indeces()
		#self._create_directories(len(self.file_indeces))
		
		for time_index in range(len(self.file_indeces)):
			print(f"Processing time index {time_index}...")
			
			self.original_dynamic_spectrum = utils._load_data(
				str(self.filename), self.file_type,
				self.file_indeces[time_index],
				self.number_of_samples_array[time_index],
				self.header, self.no_rfi_cleaning, self.no_zdot
			)
			
			self.return_dm_time = np.empty(len(self.ddplan_instance.old_ddplan_dm_step), dtype=object)
			
			for dm_step_counter, dm_step in enumerate(self.ddplan_instance.old_ddplan_dm_step):
				"""
				if dm_step_counter < 1:
					dm_bins = int(np.ceil(self.ddplan_instance.dm_boundaries[dm_step_counter]/dm_step)) + 1
				else:
					dm_bins = int(np.ceil(self.ddplan_instance.dm_boundaries[dm_step_counter]/dm_step)) - int(self.ddplan_instance.dm_boundaries[dm_step_counter-1]/dm_step) +1
				"""
				
				#self._dm_time_array = np.zeros((dm_bins, 1))
				if self.subbanded:
					self.subband_data()
					
				self.dynamic_spectrum = block_reduce(
					self.original_dynamic_spectrum,
					block_size=(1, self.ddplan_instance.old_ddplan_downsampling_factor[dm_step_counter].astype(int) * self.initial_downsampling_factor),
					func=np.mean
				)
				
				if dm_step_counter < 1:
					self._dm_time_array = utils.transform(
						self.dynamic_spectrum, self.header.ftop, self.header.fbottom,
						self.ddplan_instance.old_ddplan_downsampling_factor[dm_step_counter].astype(int) * self.initial_downsampling_factor * self.header.tsamp,
						0, self.ddplan_instance.dm_boundaries[dm_step_counter],
						self.ddplan_instance.old_ddplan_dm_step[dm_step_counter]
					).data
				else:
					self._dm_time_array = utils.transform(
						self.dynamic_spectrum, self.header.ftop, self.header.fbottom,
						self.ddplan_instance.old_ddplan_downsampling_factor[dm_step_counter].astype(int) * self.initial_downsampling_factor * self.header.tsamp,
						self.ddplan_instance.dm_boundaries[dm_step_counter - 1],
						self.ddplan_instance.dm_boundaries[dm_step_counter],
						self.ddplan_instance.old_ddplan_dm_step[dm_step_counter]
					).data
					
				#self._dm_time_array = np.hstack((self.dm_time_array, dm_time_array))
				#self._dm_time_array = self.dm_time_array[:, 1:]
				self.return_dm_time[dm_step_counter] = self.dm_time_array
			
			self.executor.submit(self.prepare_slices, time_index, self.return_dm_time)
			
			if time_index > 0:
				#self.executor.submit(self.plot_results_pipeline)
				self.run_inference_pipeline()
				
		self.run_inference_pipeline()
		#self.plot_results_pipeline()
		self.results_queue.put(None)
		self.plotting_thread.join()
		self.save_prediction_information()
		end_time = time.time()
		print('That took ', end_time-start_time)

def parse_args():
	parser = argparse.ArgumentParser(
		description="Search for radio transients by creating DM-time images and feeding them on a Mask RCNN model.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument('--no-zdot', dest='nozdot', help='Disable Z-Dot filter', action='store_true')
	parser.add_argument('--no-rfi-cleaning', dest='norficleaning', help='Disable IQRM RFI cleaning', action='store_true')
	parser.add_argument('-o', '--outdir', help='Output directory for saving the results of the search.', default=os.getcwd())
	parser.add_argument('-i','--datapath', help = 'Filename of the filterbank file to dedisperse', default=os.getcwd())
	parser.add_argument('--nchunk', type=int, help='Number of chunks the search will be split into (Number of sub-integrations for PSRF FITS file or seconds of data for filterbank file)',default=100)
	parser.add_argument('--subband',help='Top frequency and bottom frequency channel to subband the data to. This should be a comma-separated list of the values in MHz.', default=False)
	
	return parser.parse_args()
	
if __name__ == '__main__':
	args = parse_args()
	config = Config()
	transform = transforms.Compose([Normalize_DM_time_snap(),ToTensor()])
	hermes = Hermes(config, args.datapath, args.outdir, args.subband, transform, args.nchunk, args.norficleaning, args.nozdot)
	hermes.search()
	
	
