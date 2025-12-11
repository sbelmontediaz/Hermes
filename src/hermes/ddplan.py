"""
Sergio Belmonte Diaz 18/10/2023


Script to calculate the ddplan to follow when using FRB-Net.

The input is a txt file containing the following information in different columns: 

bandwidth(MHz), number_frequency_channels, sampling_time(s), frequency_at_top_of_the_band(MHz), minimum_searched_width(ms), maximum_searched_width(ms), width_factor, minimum_DM_to_search(pc cm^-3), maximum_DM_to_search(pc cm^-3)

To run the script, simply use the name of the txt file as the first argument and the name to be used to save the .png file containing the ddplan plot. For example:

run ddplan.py config_ddplan.txt ddplan_plot

The script calculates the most optimal DM range for the smallest width to search for. Based on that, the most optimal DM step is found. The DM ranges are found considering different smearing effects
induced when performing incoherent dedispersion: sampling time smearing, intrachannel smearing at the top of the band and dm step smearing. When these effects are strong enough so that a burst at a given width
would not be resolved, the new range of DM starts, for which the filterbank file gets downsampled by 2 and the DM step increases by 2. Finally, an overlap between the different DM ranges is imposed to ensure that
a burst at a limiting DM would be detected at either DM range.

The script outputs a plot containing the different smearing factors at different DMs, and it prints out the DM range and DM step.

"""
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import utils
from config import Config


class ddplan(object):

	def __init__(self,config,outfname,subbanded = False,fname=False):
		self.config = config
		self._outfname = str(outfname)
		self.subbanded = subbanded
		self.fname = fname
		if self.fname:
			self._get_info_from_header()
		self._min_width = self.config.min_width
		self._width_array = self.config.width_step**(np.arange(np.log(self.config.min_width)/np.log(self.config.width_step),np.log(self.config.max_width*self.config.width_step)/np.log(self.config.width_step))).astype(int)
		self._first_dm_step = 0.0
		self._old_ddplan_dm_min = np.zeros((1))
		self._old_ddplan_dm_max = np.zeros((1))
		self._old_ddplan_dm_step = np.zeros((1))
		self._old_ddplan_downsampling_factor = np.zeros((1))
		self._old_number_of_dms = np.zeros((1))
		self._check_min_width_and_tsamp()
		self._dm_trials = np.empty(0)
		self._dm_boundaries = np.empty(0)
	
	def _get_info_from_header(self):
		header = utils._open_header(self.fname, str(self.fname).split(".")[-1])
		"""
		if self.subbanded:
			chantop = int((self.header.ftop - self.subbanded[0])/np.abs(self.header.foff))
			chanbottom = int((self.header.ftop - self.subbanded[1])/np.abs(self.header.foff))
			self.header.subband_data(self.header.ftop-chantop*np.abs(self.header.foff),self.header.ftop-chanbottom*np.abs(self.header.foff))
		"""
		self.config.ftop		=		header.ftop
		self.config.fbottom		=		header.fbottom
		self.config.bandwidth	=		header.bandwidth
		self.config.nchans		=		header.nchans
		self.config.tsamp		=		header.tsamp
	
	"""
	@property
	def fname(self):
		return (self._fname)
	"""
	
	@property
	def dm_trials(self):
		return (self._dm_trials)
	
	@property
	def dm_boundaries(self):
		return (self._dm_boundaries)

	@property
	def outfname(self):
		return (self._outfname)

	@property
	def min_width(self):
		return (self._min_width)
	"""
	@property
	def bandwidth(self):
		return (self._bandwidth)

	@property
	def nchans(self):
		return (self._nchans)

	@property
	def tsamp(self):
		return (self._tsamp)

	@property
	def ftop(self):
		return (self._ftop)

	@property
	def min_width(self):
		return (self._min_width)

	@property
	def max_width(self):
		return (self._max_width)

	@property
	def width_step(self):
		return (self._width_step)

	@property
	def dm_min(self):
		return (self._dm_min)

	@property
	def dm_max(self):
		return (self._dm_max)

	@property
	def fbottom(self):
		return (self._fbottom)
	"""
	@property
	def width_array(self):
		return (self._width_array)

	@property
	def width_labels(self):
		return (self._width_labels)
	"""
	@property
	def KDM(self):
		return (self._KDM)

	@property
	def fudge_factor(self):
		return (self._fudge_factor)

	@property
	def window_size(self):
		return (self._window_size)
	"""
	@property
	def first_dm_step(self):
		return (self._first_dm_step)

	@property
	def old_ddplan_dm_min(self):
		return (self._old_ddplan_dm_min)

	@property
	def old_ddplan_dm_max(self):
		return (self._old_ddplan_dm_max)

	@property
	def old_ddplan_dm_min_float(self):
		return (self._old_ddplan_dm_min_float)

	@property
	def old_ddplan_dm_max_float(self):
		return (self._old_ddplan_dm_max_float)

	@property
	def old_ddplan_dm_step(self):
		return (self._old_ddplan_dm_step)
		
	@property
	def new_ddplan_dm_step(self):
		return (self._new_ddplan_dm_step)

	@property
	def old_ddplan_downsampling_factor(self):
		return (self._old_ddplan_downsampling_factor.astype(int))

	@property
	def old_number_of_dms(self):
		return (self._old_number_of_dms)

	def _check_min_width_and_tsamp(self):
		factor_width = int(math.log2(self.config.min_width))
		factor_sampling_time = int(np.ceil(np.array([math.log2(self.config.tsamp*10**6)])[0]))
		if np.abs(factor_width-factor_sampling_time) >= 2:
			self.config.tsamp = 2**(factor_width-1)*10**-6

	def DM_delay(self,ftop, fbottom, DM):
		return self.config.KDM*DM*((fbottom)**-2 - (ftop)**-2)

	def inverse_DM_delay(self, ftop, fbottom, time_delay):
		return time_delay/(self.config.KDM*((fbottom)**-2 -  (ftop)**-2))

	def sampling_time_smearing(self,DM,downsampling_factor):
		if type(DM) == int:
			return downsampling_factor*self.config.tsamp*10**6
		else:
			return downsampling_factor*self.config.tsamp*np.ones((len(DM)))*10**6

	def intrachannel_smearing(self,DM):
		return self.DM_delay(self.config.bandwidth/self.config.nchans+self.config.ftop, self.config.ftop-self.config.bandwidth/self.config.nchans,DM)*10**6

	def inverse_intrachannel_smearing(self, time_smearing):
		return self.inverse_DM_delay(self.config.bandwidth/self.config.nchans+self.config.ftop, self.config.ftop-self.config.bandwidth/self.config.nchans, time_smearing)

	def dm_step_smearing(self, DM, DM_step):
		if type(DM) == int:
			return self.DM_delay(self.config.ftop,self.config.fbottom,DM_step/2)*10**6
		else:
			return np.ones((len(DM)))*self.DM_delay(self.config.ftop,self.config.fbottom,DM_step/2)*10**6

	def native_dm_step(self):
		return self.config.tsamp / (self.config.KDM * (self.config.fbottom**-2 - self.config.ftop**-2))

	def finding_first_dm_step(self):
		optimal_dm_range = 2*(self.config.fudge_factor*(self.min_width*10**-6)) / (self.config.KDM*(self.config.fbottom**-2 - (self.config.ftop)**-2))
		optimal_dm_step = optimal_dm_range / self.config.window_size
		if optimal_dm_step < self.native_dm_step():
			self._first_dm_step = self.native_dm_step()
			resolved_width = (self.config.window_size * self.native_dm_step()) * (self.config.KDM*(self.config.fbottom**-2 - (self.config.ftop)**-2)) / (2*self.config.fudge_factor) * 10**6
			print("Warning: The minimum burst width cannot be resolved. The minimum width to be resolved in this case is ", round(resolved_width,2), "\u00B5s." )
			resolved_width_index = np.where(self.width_array > resolved_width)[0][0]
			self._width_array = self.width_array[resolved_width_index:]
			self._min_width = self.width_array[0]
			print("The ddplan will be calculated considering the closest avalaible width in your search as the first width, which is ", self.min_width,"\u00B5s.")
		else:
			self._first_dm_step = (optimal_dm_step//self.native_dm_step())*self.native_dm_step()

	def finding_dm_ranges(self):
		downsample_factor = 1
		DM_max_list = []
		DM_max_list_floats = []
		happened = False
		for width in self.width_array:
			# Find the total smearing time that would cause the width of the burst be 1.5 times the original width
			smearing_time = np.sqrt((1.5*width)**2-(width)**2) * 10**-6
			# Subtract the DM step and sampling time smearing since they are independent of DM
			smearing_time -= (self.sampling_time_smearing(0,downsample_factor) + self.dm_step_smearing(0,self.first_dm_step*downsample_factor)) * 10**-6
			if smearing_time <= 0:
				continue 
			# Find the corresponding DM that would cause an intrachannel smearing of the given value
			DM_max = self.inverse_intrachannel_smearing(smearing_time)
			if DM_max >= self.config.dm_max:
				if len(DM_max_list_floats) < 1:
					break
				if happened == False:
					if (self.config.dm_max - DM_max_list_floats[-1])//(self.first_dm_step*downsample_factor) < self.config.image_size:
						DM_max_list.pop()
						DM_max_list_floats.pop()
						happened = True
				continue
			# Increase the downsampling factor for each range
			downsample_factor *= self.config.width_step
			DM_max_list_floats.append(DM_max)
			DM_max_list.append(int(DM_max))
		DM_max_list.append(int(self.config.dm_max))
		DM_max_list_floats.append(self.config.dm_max)
		self._old_ddplan_dm_max = np.array(DM_max_list)
		self._old_ddplan_dm_max_float = np.array(DM_max_list_floats)
		self._old_ddplan_dm_min_float = np.zeros((len(self.old_ddplan_dm_max_float)))
		self._old_ddplan_dm_min = np.zeros((len(self.old_ddplan_dm_max))).astype(int)
		#Define the lower end of the DM range
		for i in np.arange(1,len(self.old_ddplan_dm_max)):
			self._old_ddplan_dm_min[i] = int(self.old_ddplan_dm_max[i-1])
			self._old_ddplan_dm_min_float[i] = self.old_ddplan_dm_max[i-1]
		self._old_ddplan_dm_min[0] = int(self.config.dm_min)
		self._old_ddplan_dm_min_float[0] = self.config.dm_min
		# Add the overlapping DM regions
		for i in range(len(self.old_ddplan_dm_min)):
			if i < len(self.old_ddplan_dm_min):
				self._old_ddplan_dm_max[i] += int(self.config.overlap*self.first_dm_step*2**i)
				self._old_ddplan_dm_max_float[i] += self.config.overlap*self.first_dm_step*2**i
			if i >0:
				self._old_ddplan_dm_min[i] -= int(self.config.overlap*self.first_dm_step*2**(i))
				self._old_ddplan_dm_min_float[i] -= self.config.overlap*self.first_dm_step*2**i 

		#self._old_ddplan_dm_max[-1] = int(self.config.dm_max)
		#self._old_ddplan_dm_max_float[-1] = self.config.dm_max
	
	def create_dm_steps_array(self):
		self._old_ddplan_dm_step = np.ones((len(self.old_ddplan_dm_min)))*self.first_dm_step*self.config.width_step**(np.arange(0,len(self.old_ddplan_dm_min))) 

	def create_downsampling_factor(self):
		self._old_ddplan_downsampling_factor = np.ones((len(self.old_ddplan_dm_min)))*self.config.width_step**(np.arange(0,len(self.old_ddplan_dm_min))).astype(int)

	def create_number_of_dms_array(self):
		self._old_number_of_dms = ((self.old_ddplan_dm_max-self.old_ddplan_dm_min)/self.old_ddplan_dm_step).astype(int) + 2

	def print_ddplan(self):
		
		print("The original dedispersion plan generated looks like the following: ")
		print("DM min\tDM max\tDM step\tNum of DMs Percentage of DMs")
		for i in range(len(self.old_ddplan_dm_min)):
			print(round(self.old_ddplan_dm_min_float[i],1),'\t',round(self.old_ddplan_dm_max_float[i],1),'\t',round(self.old_ddplan_dm_step[i],3),'\t',self.old_number_of_dms[i],'\t',round(self.old_number_of_dms[i]/self.old_number_of_dms.sum()*100,1),'%')
		print("Total number of DMs searched: ", self.old_number_of_dms.sum())
		
		print("The dedispersion plan generated looks like the following: ")
		print("DM min\tDM max\tDM step\tNum of DMs Percentage of DMs")
		for i in range(len(self.dm_trials)):
			if i < 1:
				print(self.config.dm_min,'\t',round(self.dm_boundaries[i],1),'\t',round(self.new_ddplan_dm_step[i],3),'\t',self.dm_trials[i],'\t',round(self.dm_trials[i]/self.dm_trials.sum()*100,1),'%')
			else:
				print(round(self.dm_boundaries[i-1],1),'\t',round(self.dm_boundaries[i],1),'\t',round(self.new_ddplan_dm_step[i],3),'\t',self.dm_trials[i],'\t',round(self.dm_trials[i]/self.dm_trials.sum()*100,1),'%')
		print("Total number of DMs searched: ", self.dm_trials.sum())

	def plot_ddplan(self):
		# Generate an array of DMs for each DM range
		length = 10**4
		DMs = np.zeros((len(self.old_ddplan_dm_min),length))
		for dm in range(len(self.old_ddplan_dm_min)):
			DMs[dm] = np.linspace(self.old_ddplan_dm_min[dm],self.old_ddplan_dm_max[dm],length)

		plt.figure(figsize=(12,12))
		plt.yscale('log')
		lines = []
		for i, dm_step in enumerate(self.old_ddplan_dm_step):
			# Calculate the different smearing factors and plot them
			delta_tsamp = self.sampling_time_smearing(DMs[i],self.old_ddplan_downsampling_factor[i])
			delta_chan = self.intrachannel_smearing(DMs[i])
			delta_dm_step = self.dm_step_smearing(DMs[i],dm_step)
			total_smearing = delta_tsamp + delta_chan + delta_dm_step
			line1, =  plt.plot(DMs[i],delta_tsamp+0.01, 'b')
			#plt.text(DMs[i][length//2],delta_tsamp+0.1,str(round(delta_tsamp[0]+0.1,2)))
			line2, = plt.plot(DMs[i],delta_chan, 'r')
			line3, = plt.plot(DMs[i],delta_dm_step, 'g')
			#plt.text(DMs[i][length//2],delta_dm_step,str(round(delta_dm_step[0],2)))
			line4, = plt.plot(DMs[i],total_smearing,'k')
			#print(DMs[i][length//2],delta_tsamp[0])
			lines.extend([line1, line2, line3, line4])
			plt.text(DMs[i][length//2],delta_dm_step[0],str(round(delta_dm_step[0],2)),va='bottom', fontsize='14',color='green')
			plt.text(DMs[i][length//2],delta_tsamp[0]+0.01,str(round(delta_tsamp[0]+0.01,2)),va='bottom', fontsize='14',color='blue')
		plt.xlabel("DM",fontsize=16)
		plt.ylabel("Smearing (ms)",fontsize=16)
		plt.xticks(fontsize=15)
		plt.yticks(fontsize=15)
		plt.xlim(np.min(DMs),np.max(DMs))
		plt.ylim(self.dm_step_smearing(DMs[0],self.old_ddplan_dm_step[0])[-1]/10,total_smearing[-1]*3)
		labels = ['sampling time', 'channel smearing', 'dm_step smearing', 'total smearing']
		plt.legend(lines,labels,fontsize=14)
		self._width_labels = np.core.defchararray.add(self._width_array.astype(str),'ms')
		for hline, label in zip(self.width_array,self.width_labels):
			plt.axhline(hline, linestyle='dotted', color='gray')
			plt.annotate(f"{label}", xy=(np.min(DMs),hline), color='black',verticalalignment='bottom', fontsize=13)
		plt.savefig(self.outfname+'.png',dpi=400)
		
		#Plot new ddplan
		length = 10**4
		DMs = np.zeros((len(self.old_ddplan_dm_min),length))
		for dm in range(len(self.old_ddplan_dm_min)):
			if dm < 1:
				DMs[dm] = np.linspace(0,self.dm_boundaries[dm],length)
			else:
				DMs[dm] = np.linspace(self.dm_boundaries[dm-1],self.dm_boundaries[dm],length)

		plt.figure(figsize=(12,12))
		plt.yscale('log')
		lines = []
		for i, dm_step in enumerate(self.old_ddplan_dm_step):
			# Calculate the different smearing factors and plot them
			delta_tsamp = self.sampling_time_smearing(DMs[i],self.old_ddplan_downsampling_factor[i])
			delta_chan = self.intrachannel_smearing(DMs[i])
			delta_dm_step = self.dm_step_smearing(DMs[i],dm_step)
			total_smearing = delta_tsamp + delta_chan + delta_dm_step
			line1, =  plt.plot(DMs[i],delta_tsamp+0.01, 'b')
			#plt.text(DMs[i][length//2],delta_tsamp+0.1,str(round(delta_tsamp[0]+0.1,2)))
			line2, = plt.plot(DMs[i],delta_chan, 'r')
			line3, = plt.plot(DMs[i],delta_dm_step, 'g')
			#plt.text(DMs[i][length//2],delta_dm_step,str(round(delta_dm_step[0],2)))
			line4, = plt.plot(DMs[i],total_smearing,'k')
			#print(DMs[i][length//2],delta_tsamp[0])
			lines.extend([line1, line2, line3, line4])
			plt.text(DMs[i][length//2],delta_dm_step[0],str(round(delta_dm_step[0],2)),va='bottom', fontsize='14',color='green')
			plt.text(DMs[i][length//2],delta_tsamp[0]+0.01,str(round(delta_tsamp[0]+0.01,2)),va='bottom', fontsize='14',color='blue')
		plt.xlabel("DM",fontsize=16)
		plt.ylabel("Smearing (ms)",fontsize=16)
		plt.xticks(fontsize=15)
		plt.yticks(fontsize=15)
		plt.xlim(np.min(DMs),np.max(DMs))
		plt.ylim(self.dm_step_smearing(DMs[0],self.old_ddplan_dm_step[0])[-1]/10,total_smearing[-1]*3)
		labels = ['sampling time', 'channel smearing', 'dm_step smearing', 'total smearing']
		plt.legend(lines,labels,fontsize=14)
		self._width_labels = np.core.defchararray.add(self._width_array.astype(str),'ms')
		for hline, label in zip(self.width_array,self.width_labels):
			plt.axhline(hline, linestyle='dotted', color='gray')
			plt.annotate(f"{label}", xy=(np.min(DMs),hline), color='black',verticalalignment='bottom', fontsize=13)
		plt.savefig(self.outfname+'new.png',dpi=400)
		plt.show()

	def save_arrays_in_npy(self):
		print("Saving DM ranges and DM steps numpy array into", str(self.outfname)+".npy")
		ddplan_arrays = np.zeros((3,len(self.old_ddplan_dm_min)))
		ddplan_arrays[0] = self.old_ddplan_dm_min
		ddplan_arrays[1] = self.old_ddplan_dm_max
		ddplan_arrays[2] = self.old_ddplan_dm_step
		np.save(self.outfname,ddplan_arrays)

	def calculate_ddplan(self):
		self.finding_first_dm_step()
		self.finding_dm_ranges()
		self.create_dm_steps_array()
		self.create_downsampling_factor()
		self.create_number_of_dms_array()
	
	def closest_integer_of_full_windows_next(self,num):
		full_windows = (num - self.config.image_size) / (self.config.image_size - self.config.overlap)
		closest_integer = math.ceil(full_windows)
		return closest_integer
		
	def closest_integer_of_full_windows_before(self,num):
		full_windows = (num - self.config.image_size) // (self.config.image_size - self.config.overlap)
		if full_windows < 1:
			full_windows += 1
		return full_windows
		
	def inverse_sum_powers_two(self,array):
		length = len(array)
		return np.sum(array[::-1] / (2 ** np.arange(1, length + 1)))

	def calculate_new_indeces(self):
		
		for dm_range in range(len(self.old_ddplan_dm_min)):
			if len(self.dm_trials) < 1:
				self._dm_trials = np.append(self.dm_trials, self.closest_integer_of_full_windows_next(self.old_number_of_dms[dm_range])*(self.config.image_size - self.config.overlap) + self.config.image_size)
				self._dm_boundaries = np.append(self.dm_boundaries,self.dm_trials[dm_range] * self.old_ddplan_dm_step[dm_range])
			else:
				#Calculate the amount of DM bins that would not fully fit in the windows for the previous chunk of DM trials
				remainder = self.inverse_sum_powers_two(self.dm_trials) - self.config.image_size - self.closest_integer_of_full_windows_before(self.inverse_sum_powers_two(self.dm_trials)) * (self.config.image_size - self.config.overlap)
				self._dm_trials = np.append(self.dm_trials, self.closest_integer_of_full_windows_next((self.old_ddplan_dm_max[dm_range] - self.dm_boundaries[-1])/self.old_ddplan_dm_step[dm_range] + remainder) * (self.config.image_size - self.config.overlap) + self.config.image_size - remainder)
				self._dm_boundaries = np.append(self.dm_boundaries, self.dm_trials[-1]*self.old_ddplan_dm_step[dm_range] + self.dm_boundaries[-1])
		self._new_ddplan_dm_step = self.old_ddplan_dm_step.copy()
		if self.dm_trials[-1] < self.config.image_size and self.dm_boundaries[-1] > self.config.dm_max:
			self._dm_trials = self.dm_trials[:-1]
			self._dm_boundaries = self.dm_boundaries[:-1]
			self._new_ddplan_dm_step = self.new_ddplan_dm_step[:-1]
			


def main(fname,outfname):
	ddplan_class = ddplan(fname,outfname)
	ddplan_class.calculate_ddplan()
	ddplan_class.calculate_new_indeces()
	ddplan_class.print_ddplan()
	ddplan_class.save_arrays_in_npy()
	ddplan_class.plot_ddplan()

if __name__ == '__main__':
   
	outfname = sys.argv[1]
	#outfname =  sys.argv[2]
	config = Config()	
	main(config,outfname)
