'''
    File name: pattern_calculator_fastest.py
    Author: WeiChung Chang
    Date created: 06/20/2019
    Date last modified:
    Python Version: 3.5
'''

import numpy as np
import sys
import time
import bisect

from tqdm import tqdm

from definition import patternType
from definition import patten_param_table
from base_feature_generator import generate_base_feature

from utils import pad_zeros_to_imgs
from utils import integral_array
from utils import integral_array_vertical
from utils import integral_array_horizontal

import logging, sys

# definitons
DIR               = 0
MAJOR_PATTERN_LEN = 1
EXPAND            = 2

H              = 0
W              = 1

# implementation
class TrainFeatureMapGenerator():
	"""
	A class calculates featue map from images

	Attributes
	----------
	raw_imgs : ndarray
		image set to train
	ingegral_h : ndarray
		integral img at horizotal direction
	ingegral_v : ndarray
		integral img at vertical direction
	self.h : int
		raw image height
	self.w : int
		raw image width

	Methods
	-------
	says(sound=None)
		Prints the animals name and what sound it makes
	"""
	def __init__(self, raw_imgs):
		self.pad_imgs     = pad_zeros_to_imgs(raw_imgs) # (N, H, W)
		self.ingegral_h   = integral_array_horizontal(self.pad_imgs)
		self.ingegral_v   = integral_array_vertical(self.pad_imgs)
		self.h            = raw_imgs.shape[1]
		self.w            = raw_imgs.shape[2]
		self.ph           = self.pad_imgs.shape[1]
		self.pw           = self.pad_imgs.shape[2]

		self.fmap         = []

		# private data
		self.__range_table  = []
		self.__count        =  0
		self.__min_featue_h = -1
		self.__min_featue_w = -1
		self.__max_featue_h = sys.maxsize
		self.__max_featue_w = sys.maxsize
		self.__done         = False

		self.tmp_val      = None

	def set_min_feture(self, min_h, min_w):
		"""
		set min dim of features to discard too small feature.

		Parameters
		----------
		min_h : int
		min_w : int
		"""
		self.__min_featue_h = min_h
		self.__min_featue_w = min_w

	def set_max_feture(self, max_h, max_w):
		"""
		set min dim of features to discard too small feature.

		Parameters
		----------
		min_h : int
		min_w : int
		"""
		self.__max_featue_h = max_h
		self.__max_featue_w = max_w

	def dump_range_table(self):
		"""
		Dump range table's content; for DBG
		"""
		for i, item in enumerate(self.__range_table):
			print('%dth range = [ %7d, %7d' % (i, item[0][0], item[0][1]), ") uid = ", item[1])

	def get_raw_shape(self):
		"""
		Get the shape of raw image
		"""
		return self.h, self.w

	def clean(self):
		self.fmap = []
		self.__range_table  = []
		self.__count        =  0
		self.__min_featue_h = -1
		self.__min_featue_w = -1
		self.__max_featue_h = sys.maxsize
		self.__max_featue_w = sys.maxsize
		self.__done         = False
		self.tmp_val        = None


	def prepare_to_train(self, do_argsort=True):
		"""
		After calling this function, self.fmap is 
		1. flatterned 
		2. argsorted 
		3. the content is replace by argsorted result.
		
		The users should NOT add feature anymore hereafter.
		"""
		s = time.time()
		if len(self.fmap) > 0:
			self.fmap = np.hstack(self.fmap).T
			self.tmp_val = self.fmap.copy()
			if do_argsort == True:
				self.fmap = np.argsort(self.fmap, axis=1)
			else:
				print("********** fmap has NOT been argsorted **********")
		e = time.time()
		self.__done = True

	def clear():
		self.__done = False
		self.fmap   = []

	def caculate_features_all (self):
		"""
		calculate the whole feature maps 

		"""
		s = time.time()
		for name, _ in patternType.__members__.items():
			base_features = generate_base_feature(name, (self.h, self.w))
			for bf in tqdm(base_features):
				self.caculate_features_by_base(name, bf)

		e = time.time()
		print('spent ', e - s)

		return

	def caculate_features_by_pattern (self, pattern):
		"""
		calculate feature maps provided a specific pattern

		Parameters
		----------
		pattern: string
			should be 'h', 'v' or 'd'
		"""
		base_features = generate_base_feature(pattern, (self.h, self.w))

		for bf in tqdm(base_features):
			self.caculate_features_by_base(pattern, bf)
		return

	def caculate_features_by_base (self, pattern, base):
		"""
		calculate feature maps provided a specific base

		Parameters
		----------
		pattern: string
			should be 'h', 'v' or 'd'
		base: tuple
			base shape
		"""
		dir_, pattern_major_length = patten_param_table[pattern]
		base_fmap                  = self.__caculate_base_fmap (base, dir_, pattern_major_length)
		#print('base_fmap:\n', base_fmap, '\nshape = ', base_fmap.shape, dir_, pattern_major_length)

		self.__expand_base_fmap(base_fmap, base, dir_, pattern_major_length)
		return

	def __insert_to_fmap(self, fmap, uid):
		"""
		Insert item to fmap and update range_table

		Parameters
		----------
		fmap : ndarray
			should in shape of (N, H, W)

		uid : tuple
			in the form of (dir, pattern_major_length, base) such as ('h', 3, (2, 6))
		"""
		if uid[EXPAND][H] < self.__min_featue_h or uid[EXPAND][W] < self.__min_featue_w:
			return
		if uid[EXPAND][H] > self.__max_featue_h or uid[EXPAND][W] > self.__max_featue_w:
			return
		num_of_feature = np.prod(fmap.shape[1:])

		fmap = fmap.reshape(fmap.shape[0], -1)
		self.fmap.append(fmap)
		last = self.__count + num_of_feature
		self.__range_table.append(((self.__count, last), uid))
		self.__count = last

	def __expand_base_fmap(self, base_fmap, base, dir_, pattern_major_length):
		"""
		calculate feature maps from 1.provided base feature map & 2.corresponding feature parameters.

		Parameters
		----------
		base_fmap : ndarray
			feature map of base

		base: tuple
			used to identify featue map array

		dir_: string
			should be 'h', 'v' or 'd'

		pattern_major_length : int

		"""

		n_ = base_fmap.shape[0]
		h_ = base_fmap.shape[1]
		w_ = base_fmap.shape[2]

		if dir_ == 'v':
			self.__insert_to_fmap (base_fmap, (dir_, pattern_major_length, tuple(base)))

			minor_dim = w_
			if minor_dim < 2:
				raise RuntimeError('minor_dim < 2')
			last_fmap = base_fmap
			for md in range(2, minor_dim+1):
				shift     = md - 1
				len_      = w_ - shift
				now       = base_fmap[:,:,shift:] + last_fmap[:, :, :len_]
				last_fmap = now

				self.__insert_to_fmap (now, (dir_, pattern_major_length, (base[0], md)))
		elif dir_ == 'h':
			self.__insert_to_fmap (base_fmap, (dir_, pattern_major_length, tuple(base)))

			minor_dim = h_
			if minor_dim < 2:
				raise RuntimeError('minor_dim < 2')
			last_fmap = base_fmap
			for md in range(2, minor_dim+1):
				shift     = md - 1
				len_      = h_ - shift
				now       = base_fmap[:,shift:,:] + last_fmap[:, :len_, :]
				last_fmap = now

				self.__insert_to_fmap (now, (dir_, pattern_major_length, (md, base[1])))
		else:
			minor_dim = h_
			if minor_dim < 2:
				raise RuntimeError('minor_dim < 2')
			last_fmap = base_fmap

			shift = 1
			for md in range(1, (minor_dim//2)+1):
				shift     = md
				base_len  = h_ - shift
				len_      = last_fmap.shape[1] - shift
				now       = last_fmap[:, :len_, :] - last_fmap[:,shift:,:]
				last_fmap = last_fmap[:, :base_len, :] + base_fmap[:,shift:,:]

				self.__insert_to_fmap (now, (dir_, pattern_major_length, (md*2, base[1])))
		return

	def __caculate_base_fmap (self, base, dir_, pattern_major_length):
		"""
		calculate base feature maps from 1.provided base feature map & 2.corresponding feature parameters.

		Parameters
		----------
		bese: tuple
			used to identify featue map array

		dir_: string
			should be 'h', 'v' or 'd'

		pattern_major_length : int

		"""
		if dir_ == 'h' or dir_ == 'd':
			r_h       = self.ph - 1
			r_w       = self.pw - base[0]
			basic_dim = (1, base[1]//pattern_major_length)

		else:
			r_h       = self.ph - base[1]
			r_w       = self.pw - 1
			basic_dim = (base[0]//pattern_major_length, 1)

		#logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
		#logging.debug('\nself.ingegral_h:\n', str(self.ingegral_h))
		#logging.debug('\nself.ingegral_v:\n', str(self.ingegral_v))
		#print('\nself.ingegral_h\n', self.ingegral_h)
		#print('\nself.ingegral_v\n', self.ingegral_v)

		h_l  = self.ph - basic_dim[0]
		w_l  = self.pw - basic_dim[1]
		if dir_ == 'h' or dir_ == 'd':
			basic_fmap = self.ingegral_h[:, 1:, basic_dim[1]:] - self.ingegral_h[:, 1:, 0:w_l]
		else:
			basic_fmap = self.ingegral_v[:, basic_dim[0]:, 1:] - self.ingegral_v[:, 0:h_l, 1:]

		#print('basic_fmap\n', basic_fmap)

		if pattern_major_length == 2:
			h_l  = basic_fmap.shape[1] - basic_dim[0]
			w_l  = basic_fmap.shape[2] - basic_dim[1]
		else:
			h_l  = basic_fmap.shape[1] - (2 * basic_dim[0])
			w_l  = basic_fmap.shape[2] - (2 * basic_dim[1])
			
		if dir_ == 'h' or dir_ == 'd':
			if pattern_major_length == 2:
				r = basic_fmap[:, :, basic_dim[1]:] - basic_fmap[:, :, 0:w_l]
			else:
				r = basic_fmap[:, :, basic_dim[1]*2:] - (basic_fmap[:, :, basic_dim[1]:basic_dim[1]+w_l]) + basic_fmap[:, :, 0:w_l]
		else:
			if pattern_major_length == 2:
				r = basic_fmap[:, basic_dim[0]:, :] - basic_fmap[:, 0:h_l, :]
			else:
				r = basic_fmap[:, basic_dim[0]*2:, :] - (basic_fmap[:, basic_dim[0]:basic_dim[0]+h_l, :]) + basic_fmap[:, 0:h_l, :]
		
		#print('r \n', r)
		return r		

	def __lookup_feature_index(self, feature_index):
		"""
		find corresponding item of range table which contains feature_index

		Parameters
		----------
		
		feature_index : int

		"""
		keys = [r[0] for r in self.__range_table]
		pos  = bisect.bisect_right(keys, (feature_index, sys.maxsize)) - 1

		return self.__range_table[pos]

	def __feature_offset_to_coordinate (self):
		"""
		get feature by axis of [feature_index, img_index]

		Parameters
		----------
		
		img_index : int
		
		feature_index : int
		"""
		return

	def get_feature (self, feature_index, img_index):
		"""
		get feature by axis of [feature_index, img_index]

		Parameters
		----------
		
		img_index : int
		
		feature_index : int
			
		"""
		item            = self.__lookup_feature_index(feature_index)
		off             = item[0]
		uid             = item[1]
		feature_offset  = feature_index - off[0]
		fval, coord     = self.__get_feature(uid, img_index, feature_offset)
		return uid, fval, coord


	def __get_feature (self, uid, img_offset, feature_offset):
		"""
		get feature number by
		A specific valuse within a flattened fmap could be identified by 1. ith image & 2. jth feature 

		Parameters
		----------
		uid : tuple
			in the form of (dir, pattern_major_length, base) such as ('h', 3, (2, 6))
		
		img_offsert : int
		
		feature_offset : int
			
		"""
		dir_                 = uid[DIR]
		pattern_major_length = uid[MAJOR_PATTERN_LEN]
		expand               = uid[EXPAND]

		if dir_ == 'h' or dir_ == 'd':
			base              = (1, expand[W])
			base_major_length = expand[W]
		else:
			base              = (expand[H], 1)
			base_major_length = expand[H]

		print('uid = ', uid, img_offset, feature_offset, dir_, base_major_length, expand)

		base_fmap          = self.__caculate_base_fmap (base, dir_, pattern_major_length)
		print('base_fmap = ', base_fmap.shape)
		
		shape, expand_fmap = self.__get_expand_fmap (base_fmap, dir_, base_major_length, expand)


		if img_offset > expand_fmap.shape[0]:
			print(img_offset, expand_fmap.shape[0])
			raise RuntimeError('out of range for img_offset')
		if feature_offset > expand_fmap.shape[1]:
			print(feature_offset, expand_fmap.shape[1], expand_fmap.shape, img_offset, expand_fmap.shape[0])
			raise RuntimeError('out of range for feature_offset')

		coordinate = (feature_offset//shape[1], feature_offset%shape[1]) 

		
		print('expand_fmap.shape = ', expand_fmap.shape, ' feature_offset ', feature_offset, shape[0], shape[1], 'coord ', coordinate, base, base_fmap.shape, expand_fmap[img_offset][feature_offset])

		'''
		print('np.sum(expand_fmap.shape)', np.sum(expand_fmap))
		np.savetxt('expand_fmap.txt', expand_fmap, fmt='%5.5f')
		tmp = expand_fmap[:,feature_offset]
		print('@@@@@@@ ',tmp.shape)
		tmp = np.sort(tmp)
		np.savetxt('expand_fmap_F.txt', tmp, fmt='%5.5f')
		'''

		return expand_fmap[img_offset][feature_offset], coordinate

	def __get_expand_fmap (self, base_fmap, dir_, base_major_length, target_expand):
		n_ = base_fmap.shape[0]
		h_ = base_fmap.shape[1]
		w_ = base_fmap.shape[2]

		N  = 0
		H  = 1
		W  = 2

		if dir_ == 'v':
			if target_expand == (base_major_length, 1):
				return (base_fmap.shape[H], base_fmap.shape[W]), base_fmap.reshape(base_fmap.shape[0],-1)

			minor_dim = w_
			last_fmap = base_fmap
			for md in range(2, minor_dim+1):
				shift     = md - 1
				len_      = w_ - shift
				now       = base_fmap[:,:,shift:] + last_fmap[:, :, :len_]
				last_fmap = now

				if target_expand == (base_major_length, md):
					return (now.shape[H], now.shape[W]), now.reshape(now.shape[0],-1)
		elif dir_ == 'h':
			if target_expand == (1, base_major_length):
				#return base_fmap.reshape(base_fmap.shape[0],-1)
				return (base_fmap.shape[H], base_fmap.shape[W]), base_fmap.reshape(base_fmap.shape[0],-1)

			minor_dim = h_
			last_fmap = base_fmap
			for md in range(2, minor_dim+1):
				shift     = md - 1
				len_      = h_ - shift
				now       = base_fmap[:,shift:,:] + last_fmap[:, :len_, :]
				last_fmap = now

				if target_expand == (md, base_major_length):
					return (now.shape[H], now.shape[W]), now.reshape(now.shape[0],-1)
		else:
			minor_dim = h_
			last_fmap = base_fmap

			shift = 1
			for md in range(1, (minor_dim//2)+1):
				shift     = md
				base_len  = h_ - shift
				len_      = last_fmap.shape[1] - shift
				now       = last_fmap[:, :len_, :] - last_fmap[:,shift:,:]
				last_fmap = last_fmap[:, :base_len, :] + base_fmap[:,shift:,:]

				if target_expand == (md*2, base_major_length):
					return (now.shape[H], now.shape[W]), now.reshape(now.shape[0],-1)

		return []
