import logging
import os
import sys
import time
from typing import Callable, Iterator, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import sigpyproc
import sigpyproc.readers
from numpy import typing as npt
from sigpyproc.block import FilterbankBlock
from sigpyproc.core import kernels, stats
from sigpyproc.core.rfi import RFIMask
from sigpyproc.core.stats import ChannelStats
from sigpyproc.header import Header
from utilities.py_astro_accelerate import *


def create_filterbank_block(data: np.ndarray, header: Header) -> FilterbankBlock:
    """
    Create a FilterbankBlock from a NumPy array and a Header.

    Parameters
    ----------
    data : np.ndarray
        2D array of shape (nchans, nsamples) containing the filterbank data.
    header : Header
        Header object containing metadata for the filterbank data.

    Returns
    -------
    FilterbankBlock
        A FilterbankBlock object containing the data and metadata.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a NumPy array")

    if data.ndim != 2:
        raise ValueError("Data must be a 2D array with shape (nchans, nsamples)")

    return FilterbankBlock(data, header)
"""
def read_plan_2d(
    data: np.ndarray,
    gulp: int = 16384,
    start: int = 0,
    nsamps: Optional[int] = None,
    skipback: int = 0,
    description: Optional[str] = None,
) -> Iterator:
    '''
    Custom read_plan function for 2D NumPy array to organize data into chunks.

    Parameters
    ----------
    data : np.ndarray
        A 2D numpy array of shape (nchans, nsamps), where `nchans` is the number of channels
        and `nsamps` is the number of time samples.
    gulp : int, optional
        The number of time samples to read in one go. Default is 16384.
    start : int, optional
        The starting index (sample) to read from. Default is 0.
    nsamps : Optional[int], optional
        The total number of samples to read. If None, it will read the remaining samples.
    skipback : int, optional
        Number of samples to skip back after each read. Default is 0.
    description : Optional[str], optional
        Description for progress tracking or logging.

    Returns
    -------
    Iterator
        A generator that yields chunks of the data in 1D flattened arrays.
    '''
    
    # Check if nsamps is provided, otherwise set to the remaining samples
    if nsamps is None:
        nsamps = data.shape[1] - start
    
    gulp = min(nsamps, gulp)  # Ensure gulp is not larger than the remaining data

    # Ensure skipback is within valid limits
    skipback = abs(skipback)
    if skipback >= gulp:
        raise ValueError(f"gulp ({gulp}) must be > skipback ({skipback})")

    # Calculate the number of full blocks and the remaining samples for the last block
    nreads, lastread = divmod(nsamps, (gulp - skipback))
    
    if lastread < skipback:
        nreads -= 1
        lastread = nsamps - (nreads * (gulp - skipback))

    # Create the blocks with their corresponding skipback value
    blocks = [(ii, gulp, -skipback) for ii in range(nreads)]
    
    # If there's remaining data after full blocks, create the final block
    if lastread != 0:
        blocks.append((nreads, lastread, 0))

    # Iterate over the blocks and yield the data chunks
    for ii, block, skip in blocks:
        # Slice the data to get the block of samples for all channels
        block_data = data[:, start : start + block]  # Use start to select data, not ii

        # Flatten the 2D block data into 1D and yield the block, index, and flattened data
        yield block, ii, block_data.ravel()

        # Update the start index for the next block
        start += block - skip
"""

def read_plan_2d(
	data: np.ndarray,
	gulp: int = 16384,
	start: int = 0,
	nsamps: Optional[int] = None,
	skipback: int = 0,
	description: Optional[str] = None,
) -> Iterator[Tuple[int, int, np.ndarray]]:

	if nsamps is None:
		nsamps = data.shape[1] - start
	gulp = min(nsamps, gulp)
	
	if skipback >= gulp:
		raise ValueError(f"gulp ({gulp}) must be > skipback ({skipback})")
		
	nreads, lastread = divmod(nsamps, (gulp - skipback))
	
	if lastread < skipback:
		nreads -= 1
		lastread = nsamps - (nreads * (gulp - skipback))
	
	for ii in range(nreads):
		block_data = data[:, start : start + gulp]
		yield gulp, ii, block_data.ravel(order = "F")
		start += gulp - skipback
	
	if lastread > 0:
		block_data = data[:, start : start + lastread]
		yield lastread, nreads, block_data.ravel(order="F")


# Standard value of the dispersion constant
KDM = 1.0 / 2.41e-4

# Function to downsample the data
def downsample_dmt_array(dmt_array,  dm_factor, time_factor):
		new_shape = (dmt_array.shape[0]//dm_factor, dm_factor, dmt_array.shape[1]//time_factor, time_factor)
		strides = (dmt_array.strides[0] * dm_factor, dmt_array.strides[0], dmt_array.strides[1] * time_factor, dmt_array.strides[1])
		dmt_array = np.lib.stride_tricks.as_strided(dmt_array, shape = new_shape, strides = strides)
		dmt_array = dmt_array.sum(axis=(1,3))/(time_factor*dm_factor)

#from pyfdmt: fast dm transform code from Vincent Morello. Modified to accomodate different dm steps rather than the native dm step.
class InputBlock(object):
	""" Wraps a frequency-time 2D array """
	def __init__(self, data, fch1, fchn, tsamp, dm_step):
		try:
			self._data = np.asarray(data, dtype=np.float32)
		except:
			raise ValueError("Cannot convert data to float32 numpy array")

		if not data.ndim == 2:
			raise ValueError("data must be two-dimensional")
		if not fch1 >= fchn:
			raise ValueError("The first channel must have the highest frequency")
		
		self._fch1 = float(fch1)
		self._fchn = float(fchn)
		self._tsamp = float(tsamp)
		self._dm_step = float(dm_step)

	@property
	def data(self):
		return self._data

	@property
	def tsamp(self):
		return self._tsamp

	@property
	def nchans(self):
		return self.data.shape[0]

	@property
	def nsamp(self):
		return self.data.shape[1]

	@property
	def fch1(self):
		return self._fch1

	@property
	def fchn(self):
		return self._fchn

	@property
	def foff(self):
		if self.nchans > 1:
			return (self.fchn - self.fch1) / (self.nchans - 1.0)
		else:
			return 0.0

	@property
	def freqs(self):
		return np.linspace(self.fch1, self.fchn, self.nchans)

	@property
	def delta_kdisp(self):
		""" Dispersion delay (seconds) across the band per unit of DM (pc cm^{-3}) """
		return KDM * (self.fchn**-2 - self.fch1**-2)

	@property
	def dm_step(self):
		""" Natural DM step of the fast DM transform for this data block """
		return self._dm_step #self.tsamp / self.delta_kdisp

	@property
	def original_dm_step(self):
		""" Ideal DM step size of input block used for the original pyfdmt implementation"""
		return self.tsamp / self.delta_kdisp

	@property
	def isplit(self):
		""" Line index where the tail part starts """
		if not self.nchans > 1:
			raise ValueError("Cannot split block with 1 line or less")

		# The frequency f at which the split should occur is such that:
		# f**-2 - fch1**-2 = 1/2 * (fchn**-2 - fch1**-2)
		# ie. half the total dispersion delay is accounted for between
		# fch1 and f
		f = (0.5 * self.fchn**-2 + 0.5 * self.fch1**-2) ** -0.5
		i = int((f - self.fch1) / self.foff + 0.5)
		return i

	def split(self):
		if not self.nchans > 1:
			raise ValueError("Cannot split block with 1 line or less")
		h = self.isplit
		head = InputBlock(self.data[:h], self.fch1, self.fch1 + (h-1) * self.foff, self.tsamp, self.dm_step)
		tail = InputBlock(self.data[h:], self.fch1 + h * self.foff, self.fchn, self.tsamp, self.dm_step)
		return head, tail

	def __str__(self):
		return "InputBlock(nsamp={s.nsamp}, nchans={s.nchans}, fch1={s.fch1:.3f}, fchn={s.fchn:.3f})".format(s=self)

	def __repr__(self):
		return str(self)


class OutputBlock(object):
	""" Wraps a DM-time 2D array """
	def __init__(self, input_block, ymin, ymax):
		ib = input_block
		ntrials = ymax - ymin + 1
		self._input_block = ib
		self._data = np.zeros(shape=(ntrials, ib.nsamp), dtype=np.float32)
		self._ymin = ymin
		self._ymax = ymax

	@property
	def data(self):
		""" Dedispersed data """
		return self._data

	@property
	def input_block(self):
		""" Input data block """
		return self._input_block

	@property
	def ntrials(self):
		""" Number of DM trials """
		return self.data.shape[0]

	@property
	def nsamp(self):
		""" Number of samples """
		return self.data.shape[1]

	@property
	def ymin(self):
		""" Dispersion delay (in samples) across the band for the first DM trial """
		return self._ymin

	@property
	def ymax(self):
		""" Dispersion delay (in samples) across the band for the last DM trial """
		return self._ymax

	@property
	def dm_step(self):
		""" DM step between consecutive trials """
		return self.input_block.dm_step #self.input_block.tsamp / self.input_block.delta_kdisp

	@property
	def original_dm_step(self):
		""" Ideal DM step size of input block used for the original pyfdmt implementation"""
		return self.input_block.tsamp / self.input_block.delta_kdisp

	@property
	def dm_min(self):
		""" First DM trial"""
		return self.ymin * self.dm_step
	@property
	def dm_max(self):
		""" Last DM trial """
		return self.ymax * self.dm_step

	@property
	def dms(self):
		""" List of DM trials, in the same order as they appear in the data """
		return np.linspace(self.dm_min, self.dm_max, self.ntrials)

	def plot(self, figsize=(18, 6), dpi=100):
		""" Plot the dedispersed data """
		fig = plt.figure(figsize=figsize, dpi=dpi)
		plt.imshow(
			self.data,
			# NOTE: dm_max is INclusive here
			extent=[-0.5, self.nsamp-0.5, self.dm_min - 0.5*self.dm_step, self.dm_max + 0.5*self.dm_step],
			origin='lower',
			aspect='auto'
		)
		plt.xlabel("Sample Index")
		plt.ylabel("DM trial")
		plt.tight_layout()
		return fig

	def __str__(self):
		return "OutputBlock(nsamp={s.nsamp}, ntrials={s.ntrials}, dm_min={s.dm_min:.3f}, dm_max={s.dm_max:.3f})".format(s=self)

	def __repr__(self):
		return str(self)


def transform(data, fch1, fchn, tsamp, dm_min, dm_max, dm_step):
	"""
	Compute the fast DM transform of a data block for the specified DM range

	Parameters
	----------
	data: array-like
		Two dimensional input data, in frequency-time order 
		(i.e. lines are frequency channels)
	fch1: float
		Centre frequency of first channel in data
	fchn: float
		Centre frequency of last channel in data
	tsamp: float
		Sampling time in seconds
	dm_min: float
		Minimum trial DM
	dm_max: float
		Maximum trial DM

	Returns
	-------
	out: OutputBlock
		Object that wraps a 2D array with the dedispersed data.
	"""
	if not dm_min >= 0:
		raise ValueError("dm_min must be >= 0")
	if not dm_max >= dm_min:
		raise ValueError("dm_max must be >= dm_min")

	block = InputBlock(data, fch1, fchn, tsamp, dm_step)
	original_dm_step = block.original_dm_step
	new_dm_step = block.dm_step	

	# Convert DMs to delays in samples
	ymin = int(dm_min / block.dm_step)
	ymax = int(np.ceil(dm_max / block.dm_step))
	return _transform_recursive(block, ymin, ymax,original_dm_step,new_dm_step)


def _transform_recursive(block, ymin, ymax,original_dm_step,new_dm_step):
	"""
	Transform block with a range of dispersion shifts from ymin to ymax INclusive
	"""
	out = OutputBlock(block, ymin, ymax)

	if block.nchans == 1:
		out.data[0] = block.data[0]
		return out

	### Split
	head, tail = block.split()

	### Transform
	ymin_head = int(ymin * head.delta_kdisp / block.delta_kdisp + 0.5)
	ymax_head = int(ymax * head.delta_kdisp / block.delta_kdisp + 0.5)
	thead = _transform_recursive(head, ymin_head, ymax_head,original_dm_step,new_dm_step)

	ymin_tail = int(ymin * tail.delta_kdisp / block.delta_kdisp + 0.5)
	ymax_tail = int(ymax * tail.delta_kdisp / block.delta_kdisp + 0.5)
	ttail = _transform_recursive(tail, ymin_tail, ymax_tail,original_dm_step,new_dm_step)

	### Merge
	# y = delay in samples across the whole band
	for y in range(ymin, ymax+1):
		# yh = delay across head band
		# yt = delay across tail band
		# yb = delay at interface between head and tail
		yh = int(y * head.delta_kdisp / block.delta_kdisp + 0.5)
		yt = int(y * tail.delta_kdisp / block.delta_kdisp + 0.5)
		yb = y - yh - yt

		ih = yh - thead.ymin
		it = yt - ttail.ymin
		i = y - out.ymin
		#Multiply by the integer number of the new DM step in terms of the native DM step.
		out.data[i] = thead.data[ih] + np.roll(ttail.data[it], -(yh+yb)*round(new_dm_step/original_dm_step))
	
	return out



class PSRFITS_header(object):
	def __init__(self,pri_hdr,sub_hdr):
		self.pri_hdr = pri_hdr
		self.sub_hdr = sub_hdr
		self.subband = False
		
	@property
	def tstart(self):
		return self.pri_hdr.tstart.value
	
	@property	
	def ftop(self):
		if self.subband:
			return self.new_ftop
		else:
			if self.pri_hdr.freqs.ftop.value < self.pri_hdr.freqs.fbottom.value:
				return self.pri_hdr.freqs.fbottom.value
			else:
				return self.pri_hdr.freqs.ftop.value
	
	@property
	def fbottom(self):
		if self.subband:
			return self.new_fbottom
		else:
			if self.pri_hdr.freqs.ftop.value < self.pri_hdr.freqs.fbottom.value:
				return self.pri_hdr.freqs.ftop.value
			else:
				return self.pri_hdr.freqs.fbottom.value
		
	@property
	def fcenter(self):
		return self.pri_hdr.freqs.fcenter.value
		
	@property
	def bandwidth(self):
		return self.pri_hdr.freqs.bandwidth.value
		
	@property
	def foff(self):
		if self.pri_hdr.freqs.ftop.value < self.pri_hdr.freqs.fbottom.value:
			return -self.pri_hdr.freqs.foff.value
		else:
			return self.pri_hdr.freqs.ftop.value
		
	@property
	def tsamp(self):
		return self.sub_hdr.tsamp
		
	@property
	def tobs(self):
		return self.sub_hdr.tsubint*self.sub_hdr.nsubint
		
	@property
	def nchans(self):
		if self.subband:
			return self.new_chans
		else:
			return self.sub_hdr.nchans
		
	@property
	def nsubint(self):
		return self.sub_hdr.nsubint
	
	@property
	def tsubint(self):
		return self.sub_hdr.tsubint
		
	@property
	def subint_samples(self):
		return self.sub_hdr.subint_samples
		
	@property
	def new_ftop(self):
		return self._new_ftop
	
	@property
	def new_fbottom(self):
		return self._new_fbottom
		
	@property
	def new_chans(self):
		return self._new_chans
		
	def subband_data(self,ftop,fbottom):
		self.subband = True
		self._new_ftop = ftop
		self._new_fbottom = fbottom
		self._new_chans = int((ftop-fbottom)//np.abs(self.foff))
	

def _open_header(fname, file_type):
	if file_type == "sf":
		myFil = sigpyproc.readers.PFITSReader(fname)
		return PSRFITS_header(myFil.pri_hdr, myFil.sub_hdr)
	else:
		myFil = sigpyproc.readers.FilReader(fname)
		return myFil.header

def clean_rfi(
	data: npt.ArrayLike,
	header: Header,
    method: str = "mad",
    threshold: float = 3,
    chanmask: Union[npt.ArrayLike, None] = None,
    custom_funcn: Union[Callable[[npt.ArrayLike], np.ndarray], None] = None,
) -> Tuple[np.ndarray, RFIMask]:
    """Clean RFI from the data.

    Parameters
    ----------
    method : str, optional
        method to use for cleaning ("mad", "iqrm"), by default "mad"
    threshold : float, optional
        threshold for cleaning, by default 3
    chanmask : :py:obj:`~numpy.typing.ArrayLike`, optional
        User channel mask to use (1 or True for bad channels), by default None
    custom_funcn : :py:obj:`~typing.Callable`, optional
        Custom function to apply to the mask, by default None
    data : npt.ArrayLike
        2D NumPy array with shape (nchans, nsamples)
    header : Header
        The observational metadata
    Returns
    -------
    Tuple[np.ndarray, RFIMask]
        Cleaned data and the RFI mask

    Raises
    ------
    ValueError
        If `method` is not "mad" or "iqrm"
    """
    if chanmask is None:
        chanmask = np.zeros(header.nchans, dtype="bool")
    if method not in {"mad", "iqrm"}:
        raise ValueError("Clean method must be 'mad' or 'iqrm'")

    # 1st pass to compute channel statistics (upto kurtosis)
    chan_stats = compute_stats(data,header)

    assert isinstance(chan_stats, stats.ChannelStats)
    # Initialise mask
    rfimask = RFIMask(
        threshold,
        header,
        chan_stats.mean,
        chan_stats.var,
        chan_stats.skew,
        chan_stats.kurtosis,
        chan_stats.maxima,
        chan_stats.minima,
    )
    rfimask.apply_mask(chanmask)
    rfimask.apply_method(method)
    if custom_funcn is not None:
        rfimask.apply_funcn(custom_funcn)

    maskvalue = np.median(data[rfimask.chan_mask,::])
    # Apply the channel mask
    masked_data = apply_channel_mask(
        data, header, rfimask.chan_mask, maskvalue
    )
    return masked_data
    

def compute_stats(data, header) -> None:
    """Compute channelwise statistics of data (upto kurtosis).

    Parameters
    ----------
    **plan_kwargs : dict
        Keyword arguments for :func:`read_plan`.
    """
    bag = stats.ChannelStats(header.nchans, header.nsamples)
    for nsamps, ii, data in read_plan_2d(data):
        bag.push_data(data, nsamps, ii, mode="full")
    return bag
    
def apply_channel_mask(
    data: npt.ArrayLike,
    header: Header,
    chanmask: npt.ArrayLike,
    maskvalue: Union[int, float] = 0,
    **plan_kwargs,
) -> str:
    """Apply a channel mask to the data.

    Parameters
    ----------
    data : npt.ArrayLike
        2D NumPy array with shape (nchans, nsamples)
    header : Header
        The observational metadata
    chanmask : :py:obj:`~numpy.typing.ArrayLike`
        boolean array of channel mask (1 or True for bad channel)
    maskvalue : int or float, optional
        value to set the masked data to, by default 0
    **plan_kwargs : dict
        Additional keyword arguments for :func:`read_plan`.

    Returns
    -------
    str
        name of output file
    """
    mask = np.array(chanmask).astype("bool")
    maskvalue = np.float32(maskvalue).astype(header.dtype)
    for nsamps, _ii, datachunk in read_plan_2d(data):
        kernels.mask_channels(datachunk, mask, maskvalue, header.nchans, nsamps)
    return data

def _load_data(fname, file_type, starting_sample, number_samples, header, no_rfi_cleaning=False, no_zdot=False, use_aa_sigproc=False):
	"""
	Load a block of data from a radio observation and perform RFI cleaning. The RFI cleaning can be zdot and IQRM.

	Parameters
	----------
	fname : str
		The path to the observation file
	file_type : str
		A string that can be either sf or fil, specifying if the observation is a filterbank file or a PSRFITS file
	starting_sample : int
		Starting sample of the block to extract. If a PSRFITS, starting subintegration index
	number_samples : int
		Number of samples to extract. If PSRFITS, number of subints to extract
	header : Header
        The observational metadata
    no_rfi_cleaning : bool
    	Optional flag if no rfi cleaning is wanted
    no_zdot : bool
    	Optional flag if no zdot filter is wanted

	Returns
	-------
	clean_data : ndarray
		Clean data with the same shape as input
	"""
	if file_type == "sf":
		myFil = sigpyproc.io.pfits.PFITSFile(fname)
		try:
			myBlock = myFil.read_subints(starting_sample,number_samples).T
		except IndexError:
			print("The number of samples given exceeds the length of the file!")
			myBlock = myFil.read_subints(starting_sample,header.nsubint-starting_sample).T
		myFil = sigpyproc.readers.PFITSReader(fname)
		myFil.header.nsamples = myBlock.T.shape[1]
		myFil.header.fch1 = header.ftop
		myFil.header.foff = header.foff
		if not no_zdot:
			myBlock = zdot(myBlock)
		if not no_rfi_cleaning:
			myBlock = clean_rfi(data=myBlock, header=myFil.header, method='iqrm')
		
		return myBlock
	else:
		# TODO: Remove the dependency on astro-accelerate and write a class to read the filterbank file
		# using the any C++ library so that we get the functionality to read the filterbank file
		# from anywhere in between the file. For now, the chunks need to be divided separately
		if use_aa_sigproc:
			with aa_py_sigproc_input(fname) as sigproc_input:
				metadata = sigproc_input.read_metadata()
				sigproc_input.read_signal()
				buffer = sigproc_input.input_buffer()
				myBlock = np.ctypeslib.as_array(buffer, shape=(int(header.nchans*header.nsamples),)).astype(np.uint8)
				myBlock = myBlock.reshape((int(header.nchans), int(header.nsamples)))
		else:
			myFil = sigpyproc.readers.FilReader(fname)
			try:
				myBlock = myFil.read_block(starting_sample, number_samples)
			except ValueError:
				myBlock = myFil.read_block(starting_sample, header.nsamples-starting_sample)
		if not no_zdot:
			myBlock = zdot(myBlock)
		if not no_rfi_cleaning:
			myBlock = clean_rfi(data=myBlock, header=myFil.header, method='iqrm')
		
		return myBlock

def zdot(data):
	"""
	Apply the Z-dot filter to the data out of place, return a cleaned copy. 
	The Z-dot filter is a much better variant of the zero-DM filter.
	See Men et al. 2019:
	https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.3957M/abstract

	Parameters
	----------
	data : ndarray
		2D array with the data in FT order, with shape (nchan, nsamp)

	Returns
	-------
	clean_data : ndarray
		Clean data with the same shape as input
	"""
	z = data.sum(axis=0)
	z -= z.mean()
	z /= np.sum(z**2) ** 0.5
	w = (data * z).sum(axis=1)
	data -= w.reshape(-1, 1) * z
	return data




