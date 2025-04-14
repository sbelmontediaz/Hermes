import numpy as np
import matplotlib.pyplot as plt
import csv
import glob
import sigpyproc,sigpyproc.readers
import sys,os
import h5py
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.models.segmentation
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.colors as mcolors
from config import Config
import utils
from pathlib import Path
from tqdm import tqdm
import time
import functools as ft
from skimage.measure import block_reduce
from astropy.stats import sigma_clipped_stats
import argparse

from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

#List of colours to produce the plots.
color_list = np.array(['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'])
#Headers of the information saved in the csv file.
column_names = ['ID','DM_time_range','width_(\u00B5s)', 'time_(s)', 'DM', 'Prediction_score' ,'Burst in image', 'slice_num','rows_in_slice','cols_in_slice','filterbank_file','time_inside_filterbank','ml_prediction_image_name']

np.random.seed(10)

#Class to perform data processing of the DM-time transform. It performs the downsampling, slicing and normalisation.
class Data_processing(object):

	def __init__(self, config, data_path):
		self.config = config
		self.data_path = data_path
		self.dm_time_array = np.zeros((1,1))
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

	#Function to access the DM-time transform saved in the .hdf5 file.
	def obtain_dm_time_array(self, group, index):
		with h5py.File(self.data_path,"r") as f:
			print('Loading the DM-time array')
			start_time = time.time()
			self.dm_time_array = f[str(group)][str(index)][...]
			end_time = time.time()
			print("That took ", round(end_time-start_time,4), " s.")
			self._loading_time += end_time-start_time

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
	def downsample_dm_time_array(self, dm_factor, time_factor):
		new_shape = (self.dm_time_array.shape[0]//dm_factor, dm_factor, self.dm_time_array.shape[1]//time_factor, time_factor)
		strides = (self.dm_time_array.strides[0] * dm_factor, self.dm_time_array.strides[0], self.dm_time_array.strides[1] * time_factor, self.dm_time_array.strides[1])
		self.dm_time_array = np.lib.stride_tricks.as_strided(self.dm_time_array, shape = new_shape, strides = strides)
		self.dm_time_array = self.dm_time_array.sum(axis=(1,3))/(time_factor*dm_factor)	
		
	#Function to slice the array and normalise the slices by scaling the set to have zero mean and std of unity.
	def slice_and_normalize(self,window_size):
		print("Slicing the DM-time array.")
		start_time = time.time()
		self.slices_dm_time_array = self._slice_2d_array_into_slices(self.dm_time_array.copy(),window_size,self.config.overlap)
		self.slices_num_of_rows, self.slices_num_of_cols = self.slices_dm_time_array.shape[0:2]
		end_time = time.time()
		print("That took ", round(end_time-start_time,4), " s.")
		self._slicing_time += end_time - start_time
		print("Normalising the DM-time array.")
		start_time = time.time()
		self.mean_set = np.mean(self.slices_dm_time_array)
		self.std_set = np.std(self.slices_dm_time_array)
		#self.mean_set, median, self.std_set = sigma_clipped_stats(self.slices_dm_time_array,sigma=3.0) #self.std_set = np.std(self.slices_dm_time_array)
		#print(self.mean_set,self.std_set)
		self.slices_dm_time_array -= self.mean_set
		self.slices_dm_time_array /= self.std_set
		self.slices_dm_time_array = self.slices_dm_time_array.reshape(self.slices_num_of_rows*self.slices_num_of_cols,window_size,window_size)
		end_time = time.time()
		print("That took ", round(end_time-start_time,4), " s.")
		self._normalising_time += end_time - start_time

		
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

#Main class to perform the search.
class search_for_single_pulses(object):
	def __init__(self, config, filename, data_path, output_directory, downsampling_factor,transforms):
		self.config = config
		#self.filterbank_directory = filterbank_directory
		self.filename = Path(filename)#glob.glob(self.filterbank_directory)#+'*.fil')
		self.file_type = str(self.filename).split(".")[-1]
		self.data_path = str(Path(data_path))
		self.output_directory = str(Path(output_directory))
		self.original_downsample_factor = int(downsampling_factor)
		self.data_processing = Data_processing(config,self.data_path)
		self.transforms = transforms
		self.number_of_groups, self.number_of_dm_time_arrays = self._find_number_groups_and_dm_time_arrays()
		self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
		self.model = self._initialize_model()
		self._obtain_telescope_configuration()
		self._create_directories()
		self.predictions = []
		self.batch_index = 0
		self.group_index = 0
		self.dm_time_index = 0
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
		self.array_rows = []
		self.array_cols = []
		self.processing_time = 0
		self.plot_time = 0
		self.redownsample_factor = 1
		self.counter = 0
		self.allocated_memory = 0
		self.reserved_memory = 0
		self.max_reserved_memory = 0	
		self.inference_time = 0	
	
	#Get an attribute from the hdf5 file containing the data.
	def _get_attribute(self,attribute_name):
		with h5py.File(self.data_path,"r") as f:
			return f[str(self.group_index)].attrs[attribute_name]

	#Find the number of filterbank file groups and DM ranges in the hdf5 file.
	def _find_number_groups_and_dm_time_arrays(self):
		with h5py.File(self.data_path,"r") as f:
			num_of_groups = len(f.keys())
			num_of_dm_time_arrays = len(f["0"].keys())
		return num_of_groups, num_of_dm_time_arrays
		
	def _obtain_telescope_configuration(self):
		#Get the specific telescope configuration from the filterbank file.
		header = utils._open_header(str(self.filename),self.file_type)
		self.config.bandwidth =   header.bandwidth
		self.config.ftop =        header.ftop
		self.config.fbottom =     header.fbottom
		self.config.nchans =      header.nchans
		self.config.tsamp =       header.tsamp
		
	def _create_directories(self):
		group_number, dataset_number = self._find_number_groups_and_dm_time_arrays()
		for number in range(group_number):
			if not os.path.exists(str(self.output_directory / Path(str(number)))):
				os.makedirs(str(self.output_directory / Path(str(number))))
			
	
	#Print the memory allocated to the GPU to trace memory behaviour.
	def _print_gpu_memory_usage(self):
		self.allocated_memory = torch.cuda.memory_allocated() / 1024**2
		self.reserved_memory = torch.cuda.memory_reserved() / 1024**2
		self.max_reserved_memory = torch.cuda.max_memory_reserved() / 1024**2

		print(f"Allocated memory: {self.allocated_memory:.2f} MB")
		print(f"Reserved memory: {self.reserved_memory:.2f} MB")
		print(f"Max reserved memory: {self.reserved_memory:.2f} MB")

		print(torch.cuda.memory_summary())

	#Load the Mask RCNN model from the torchvision library to the GPU and set it to inference mode.
	def _initialize_model(self):
		torch.hub.set_dir(self.config.resnet_weights_directory)
		model=torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, min_size=self.config.image_size,max_size=self.config.image_size)
		in_features = model.roi_heads.box_predictor.cls_score.in_features
		model.roi_heads.box_predictor=FastRCNNPredictor(in_features,num_classes=self.config.num_of_classes)
		model.load_state_dict(torch.load(self.config.weights_directory))
		model.to(self.device)
		model.eval()
		return model
		
	#Function to calculcate which predictions have a score above the threshold.
	def _finding_score_values(self,dictionary, value=0.9, key='scores'):
		if len(dictionary[key])> 0:
			if dictionary[key][0].item()>=value:
				return True
			else:
				return False
		else:
			return False
			
	#Function that returns the first argument if the second one is negative.
	def _positive_or_largest(self,a,b):
		if b>0:
			return b
		else:
			return a
	"""
	def process_dm_time_array(self):
		start_processing = time.time()
		self.data_processing.downsample_dm_time_array(self.config.width_step,self.config.width_step)
		self.data_processing.slice_and_normalize(self.config.image_size)
		if self.data_processing.slices_num_of_rows !=0:	
			self.number_of_images += self.data_processing.slices_dm_time_array.shape[0]
			self.num_imgs.append(self.data_processing.slices_dm_time_array.shape[0])
			self.num_rows.append(self.data_processing.slices_num_of_rows)
			self.num_cols.append(self.data_processing.slices_num_of_cols)
			dataset = DataLoader(MyDataset(self.data_processing.slices_dm_time_array, self.transforms),batch_size=self.config.batch_size,shuffle=False,num_workers=8, pin_memory=True, prefetch_factor=256)
			end_processing = time.time()
			self.processing_time += end_processing - start_processing
			self.inference(dataset)
		else:
			end_processing = time.time()
			self.processing_time += end_processing - start_processing
	"""
			
	#Function to run inference on the Mask RCNN model and collect the results.
	def inference(self,dataset):
		self.batch_index = 0
		with torch.no_grad():
			for i, (image,label) in enumerate(tqdm(dataset,desc="Running inference.",leave=True)):
				image = image.to(self.device)
				start_time = time.time()
				self.predictions = self.model(image)
				end_time = time.time()
				self.inference_time += end_time - start_time
				
				#Check which predictions from the batch have a score above the threshold.
				good_predictions = list(map(ft.partial(self._finding_score_values,value=self.config.threshold,key='scores'),self.predictions))
				if np.any(good_predictions) == True:
					self.prediction_idx = np.where(np.array(good_predictions) == True)[0]
					start_time = time.time()
					#Call the function to plot and save the results from the inference.
					self.process_and_plot_predictions()
					end_time = time.time()
					self.plot_time += end_time - start_time
				#Keep track of the batch index to relate the prediction to the original filterbank file.
				if self.batch_index + self.config.batch_size <= self.data_processing.slices_dm_time_array.shape[0]:
					self.batch_index += self.config.batch_size
				else:
					self.batch_index += self.data_processing.slices_dm_time_array.shape[0] - self.config.batch_size
				
					
	#Main function to peform the search.
	def Mask_RCNN_inference(self):
		for group_index in range(self.number_of_groups):
			#Reset all the variables to keep track of the groups, downsampling factors, etc.
			print("Searching for bursts in group number ", group_index)
			self.counter = 0
			self.group_index = group_index
			self.downsample_factor = self.original_downsample_factor
			self.redownsample_factor = 1
			self.prediction_id = 0
			self.processing_time = 0
			self.plot_time = 0
			#self.inference_time =0
			self.prediction_information_to_csv = np.zeros((1,13))
			self.dm_time_index = 0
			print("DM-TIME INDEX ",self.dm_time_index)
			self.width = self.config.width_array[self.dm_time_index]
			print("Searching for bursts of width ", self.width, " \u00B5s.")
			#Obtain a DM-time array, slice it and load it to the dataloader.
			start_processing = time.time()
			self.data_processing.obtain_dm_time_array(group_index,self.dm_time_index)
			self.array_rows.append(self.data_processing.dm_time_array.shape[1])
			self.array_cols.append(self.data_processing.dm_time_array.shape[0])
			self.data_processing.slice_and_normalize(self.config.image_size)
			self.number_of_images += self.data_processing.slices_dm_time_array.shape[0]
			self.num_imgs.append(self.data_processing.slices_dm_time_array.shape[0])
			self.num_rows.append(self.data_processing.slices_num_of_rows)
			self.num_cols.append(self.data_processing.slices_num_of_cols)
			dataset = DataLoader(MyDataset(self.data_processing.slices_dm_time_array, self.transforms),batch_size=self.config.batch_size,shuffle=False,num_workers=8, pin_memory=True, prefetch_factor=256)
			end_processing = time.time()
			self.processing_time += end_processing - start_processing
			#Run inference and collect results.		
			self.inference(dataset)
			for dm_time_index in range(1,self.number_of_dm_time_arrays):
				start_processing = time.time()
				self.dm_time_index = dm_time_index
				print("DM-TIME INDEX ",self.dm_time_index)
				self.width = self.config.width_array[self.dm_time_index]
				print("Searching for bursts of width ", self.width, " \u00B5s.")
				#Downsample the current DM-array and append the next DM-time array.
				self.data_processing.downsample_and_merge(group_index,dm_time_index)
				self.downsample_factor *= 2
				self.array_rows.append(self.data_processing.dm_time_array.shape[1])
				self.array_cols.append(self.data_processing.dm_time_array.shape[0])
				self.data_processing.slice_and_normalize(self.config.image_size)
				self.number_of_images += self.data_processing.slices_dm_time_array.shape[0]
				self.num_imgs.append(self.data_processing.slices_dm_time_array.shape[0])
				self.num_rows.append(self.data_processing.slices_num_of_rows)
				self.num_cols.append(self.data_processing.slices_num_of_cols)
				dataset = DataLoader(MyDataset(self.data_processing.slices_dm_time_array, self.transforms),batch_size=self.config.batch_size,shuffle=False,num_workers=8, pin_memory=True, prefetch_factor=256)
				end_processing = time.time()
				self.processing_time += end_processing - start_processing
				#print(self.processing_time)
				self.inference(dataset)
			#After all the arrays in dedispersed at different DM ranges have been searched for in their corresponding widths, finish searching all the widths specified in the config file.
			for search_width in self.config.width_array[self.dm_time_index+1:]:
				start_processing = time.time()
				self.width = search_width
				print("Searching for bursts of width ", self.width, " \u00B5s.")
				self.data_processing.downsample_dm_time_array(2,2)
				#self.downsample_factor *= 2
				self.redownsample_factor *= 2
				self.array_rows.append(self.data_processing.dm_time_array.shape[1])
				self.array_cols.append(self.data_processing.dm_time_array.shape[0])
				self.data_processing.slice_and_normalize(self.config.image_size)
				self.number_of_images += self.data_processing.slices_dm_time_array.shape[0]
				self.num_imgs.append(self.data_processing.slices_dm_time_array.shape[0])
				self.num_rows.append(self.data_processing.slices_num_of_rows)
				self.num_cols.append(self.data_processing.slices_num_of_cols)
				dataset = DataLoader(MyDataset(self.data_processing.slices_dm_time_array, self.transforms),batch_size=self.config.batch_size,shuffle=False,num_workers=8, pin_memory=True, prefetch_factor=256)
				end_processing = time.time()
				self.processing_time += end_processing - start_processing
				#print(self.processing_time)
				self.inference(dataset)
			print("Done. Number of images processed:", self.number_of_images)
			#Create a csv file saving the information of the bursts detected.
			self.save_prediction_information()
	
	#Save the information of the candidates in a csv file to inspect later.
	def save_prediction_information(self):
		data_dict_list = [dict(zip(column_names,row)) for row in self.prediction_information_to_csv[1::]]
		with open(self.output_directory+"/"+str(self.group_index)+"/table_detections.csv", mode='w', newline ='') as file:
			writer = csv.DictWriter(file, fieldnames = column_names)
			writer.writeheader()
			writer.writerows(data_dict_list)
			
	#Function to plot the burst candidates and save key information.
	def process_and_plot_predictions(self):
		window_idx = np.where(self.config.width_array == self.width)[0][0]
		#Find how many candidates in each slice in the batch have a score above the threshold.
		for candidate in self.prediction_idx:
			number_of_good_candidates = 0
			for batch in range(len(self.predictions[candidate]["scores"])):
				if self.predictions[candidate]["scores"][batch].item() > self.config.threshold:
					number_of_good_candidates += 1
				else:
					pass
			#If there are too many candidates in the same slice, collect only the 9 with the highest score.
			if number_of_good_candidates > 9:
				number_of_good_candidates = 9
			#Obtain the slice and normalise it to plot it.
			image = self.data_processing.slices_dm_time_array[candidate+self.batch_index]
			image *= 255/12
			image += 255*4/12
			plt.figure(figsize=(8,8))
			plt.imshow(image,origin="lower",aspect='auto',vmin=0,vmax=255)
			boxes = self.predictions[candidate]['boxes'][0:number_of_good_candidates].detach().cpu().numpy()
			binary_mask = np.zeros((self.config.image_size,self.config.image_size))
			for batch in range(number_of_good_candidates):
				#Obtain the segmentation mask and apply a threshold to convert it to a binary mask.
				msk=self.predictions[candidate]['masks'][batch,0].detach().cpu().numpy()
				msk_nan = np.where(msk>0.5,msk,np.nan)
				binary_mask += np.where(msk<=0.5,msk,1) #Add all the segmentation masks together so that the DM-time slice can be seen for images with multiple candidates in the same slice.
				#Define the DM and time location of the candidate as the point inside the segmentation mask in which the S/N is max.
				DM, time = np.where((image-np.min(image))*binary_mask==np.max((image-np.min(image))*binary_mask))
				#THIS LINE PLOTS THE X plt.plot(time,DM,"kx",markersize=3)	
				DM = DM[0]
				time = time[0]
				#rows_id_image = (candidate+self.batch_index)//self.data_processing.slices_num_of_cols
				#cols_id_image = (candidate+self.batch_index)%self.data_processing.slices_num_of_cols
				rows_id_image = (candidate+self.batch_index)//self.data_processing.slices_num_of_cols
				cols_id_image = (candidate+self.batch_index)-rows_id_image*self.data_processing.slices_num_of_cols
				#Calculate the DM and time based on which slice the candidate is found.
				"""
				print("DM ",DM)
				print("image_size - overlap ",self.config.image_size-self.config.overlap)
				print("rows_id_image ",rows_id_image)
				print("dm_step ", self._get_attribute("dm_step_"+str(self.dm_time_index)))
				print("redownsample_factor ", self.redownsample_factor)
				print("dm_range ", self._get_attribute('dm_range_'+str(self.dm_time_index))[0])
				"""
				DM = ((DM + (self.config.image_size-self.config.overlap)*rows_id_image)*self._get_attribute("dm_step_"+str(self.dm_time_index))*self.redownsample_factor) #+ self._get_attribute('dm_range_'+str(self.dm_time_index))[0]
				#print(DM)
				#print("BURST FOUND AT ROW ",str(rows_id_image)," AND COLUMN ",str(cols_id_image)," AND SLICE ",candidate+self.batch_index)
				time=((self.config.image_size-self.config.overlap)*cols_id_image+time)*self.config.tsamp*self.redownsample_factor*self.downsample_factor
				#Collect other information for the summary of the search file.
				filterbank_files_information = self._get_attribute('files_dedispersed')[np.digitize(time,np.cumsum(self._get_attribute('length_per_file')))]
				#print('TIME:  ',time)
				#print('TIME ARRAY:   ',self._get_attribute('length_per_file'))
				#print('CUMSUM TIME ARRAY:   ',np.cumsum(self._get_attribute('length_per_file')))
				#self._positive_or_largest(time,time - np.cumsum(self._get_attribute('length_per_file'))[np.digitize(time,np.cumsum(self._get_attribute('length_per_file')))-1])
				self.prediction_information = np.array([self.prediction_id, self.dm_time_index, self.width, time, DM, self.predictions[candidate]['scores'][batch].item(),number_of_good_candidates, candidate, self.data_processing.slices_num_of_rows, self.data_processing.slices_num_of_cols, filterbank_files_information, time + self._get_attribute('total_length')*self.group_index,self.output_directory+str(self.group_index)+"/"+str(self.counter)+"_"+str(self.width)+"_"+str(round(DM,2))+"_"+str(round(time,3))+"_"+str(candidate+self.batch_index)+'.png'])
				self.prediction_information_to_csv = np.vstack((self.prediction_information_to_csv,self.prediction_information))
				self.prediction_id += 1
				#Plot the boundary box of the candidate as horizontal lines.
				plt.hlines(boxes[batch][1],boxes[batch][0],boxes[batch][2],colors = mcolors.TABLEAU_COLORS[color_list[batch]],linestyles='dashed',label=round(self.predictions[candidate]['scores'][batch].item(),4))
				plt.hlines(boxes[batch][3],boxes[batch][0],boxes[batch][2],colors = mcolors.TABLEAU_COLORS[color_list[batch]],linestyles='dashed')
				plt.vlines(boxes[batch][0],boxes[batch][1],boxes[batch][3],colors = mcolors.TABLEAU_COLORS[color_list[batch]],linestyles='dashed')
				plt.vlines(boxes[batch][2],boxes[batch][1],boxes[batch][3],colors = mcolors.TABLEAU_COLORS[color_list[batch]],linestyles='dashed')
			#Plot the segmentation mask and add axes and legend.
			#THIS LINE PLOTS THE MASK plt.imshow(binary_mask,alpha=0.4,origin="lower",aspect='auto')
			lower_DM = ((self.config.image_size-self.config.overlap)*rows_id_image)*self._get_attribute("dm_step_"+str(self.dm_time_index))*self.redownsample_factor
			upper_DM = ((self.config.image_size-self.config.overlap)*(rows_id_image+1))*self._get_attribute("dm_step_"+str(self.dm_time_index))*self.redownsample_factor
			lower_time = -(self.redownsample_factor*self.config.tsamp*self.downsample_factor*self.config.image_size)/2
			upper_time = -lower_time
			plt.xticks([0,self.config.image_size//2,self.config.image_size],[round(lower_time,2),0,round(upper_time,2)],fontsize=15)
			plt.yticks([0,self.config.image_size//2,self.config.image_size],[round(lower_DM,2),round(lower_DM + (upper_DM-lower_DM)/2,2),round(upper_DM,2)],fontsize=15)
			plt.xlabel("Time range (s)",fontsize=15)
			plt.ylabel("DM (pc cm$^{-3}$)",fontsize=15)
			plt.legend(fontsize=15)
			plt.tight_layout()
			plt.savefig(str(self.output_directory / Path(str(self.group_index)))+"/"+str(self.counter)+"_"+str(self.width)+"_"+str(round(DM,2))+"_"+str(round(time,3))+"_"+str(candidate+self.batch_index)+'.png',dpi=400)
			self.counter += 1
			#plt.show()
			#plt.clf()
			#plt.close('all')

def parse_args():
	parser = argparse.ArgumentParser(
		description="Perform search for bursts on DM - time slices using a MaskRCNN model.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument('-d', '--downsample', type=int, help='Original downsample factor applied in the time bins of the original file', default=1)
	parser.add_argument('-f', '--fpath', help='Filename of the filterbank file to search', default=os.getcwd())
	parser.add_argument('-o', '--outdir', help='Output directory for saving the candidates', default=os.getcwd())
	parser.add_argument('-i','--datapath', help = 'Filename of the hdf5 file containing DM-time slices', default=os.getcwd())
	return parser.parse_args()

if __name__ == '__main__':
	args = parse_args()
	config = Config()
	#data_path = sys.argv[1]
	#filterbank_directory = sys.argv[2]
	#output_directory = sys.argv[3]
	#downsampling_factor = sys.argv[4]
	transform = transforms.Compose([Normalize_DM_time_snap(),ToTensor()])
	search = search_for_single_pulses(config,args.fpath,args.datapath,args.outdir,args.downsample,transform)
	start_time = time.time()
	search.Mask_RCNN_inference()
	end_time = time.time()
	print('That took ', round(end_time-start_time,4), ' s.')
    


