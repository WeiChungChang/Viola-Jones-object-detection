'''
    File name: weak_classifier_trainer.py
    Author: WeiChung Chang
    Date created: 06/25/2019
    Date last modified:
    Python Version: 3.5
'''

import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import sys
import logging, sys

from weight import Weights

class WeakClassfier():

	def __init__(self, feature_idx, image_idx, parity, threshold):
		self.parity        = parity
		self.threshold     = threshold

		self.__feature_idx = feature_idx
		self.__image_idx   = image_idx


class WeakClassifierTrainer():
	def __init__ (self, fmap,  weights, labels, tmp_val=None, name='wt', argsorted=True):
		self.raw          = None
		self.fmap         = fmap
		self.weights      = weights
		self.labels       = labels
		self.used_feature = set()
		
		self.tmp_val      = tmp_val

		self.weights.w    = self.weights.w.astype(np.float64)

		if argsorted == False:
			self.raw  = fmap.copy()
			self.fmap = self.fmap = np.argsort(fmap, axis=1)

	def train (self, vobesome = False):
		f_num          = self.fmap.shape[0]
		train_data_num = self.fmap.shape[1]

		max_idx   = -1
		min_idx   = -1
		MIN       = sys.float_info.max
		MAX       = sys.float_info.min
	
		threshold =  0
		signed_w  =  self.weights.w * ((2 * self.labels) - 1)

		total_neg = np.abs(np.sum(self.weights.w * (self.labels - 1)))
		total_pos =        np.sum(self.weights.w * self.labels)

		for i in range(f_num):
			if i in self.used_feature:
				continue
			signed_ranked_w = signed_w[self.fmap[i]]
			forward         = np.cumsum(signed_ranked_w)
			min_f           = np.min(forward)
			max_f           = np.max(forward)
			parity          = 0

			if min_f < MIN:
				MIN     = min_f
				min_idx = i
			if max_f > MAX:
				MAX     = max_f
				max_idx = i

		if (total_pos - MAX) < (total_neg + MIN):
			f_idx   = max_idx
			parity  = -1
			epsilon = (total_pos - MAX)
		else:
			f_idx   = min_idx
			parity  = 1
			epsilon = (total_neg + MIN)

		signed_ranked_w = signed_w[self.fmap[f_idx]]
		ranked_labels   = self.labels[self.fmap[f_idx]] 
		forward         = np.cumsum(signed_ranked_w)

		mask = np.zeros(ranked_labels.size).astype(int)

		if parity == -1:
			img_idx = np.argmax(forward)
			mask[img_idx+1:]  = 1
		else:
			img_idx = np.argmin(forward)
			mask[0:img_idx+1] = 1
		print('img_idx = ', img_idx)
		img_idx = self.fmap[f_idx][img_idx]

		correct                      = np.logical_xor(mask, ranked_labels).astype(int)
		beta                         = (epsilon / (1 - epsilon))
		multiplier                   = np.power(beta, correct)
		self.weights.w[self.fmap[i]] = self.weights.w[self.fmap[i]] * multiplier
		
		#self.used_feature.add((f_idx, img_idx, parity))
		self.used_feature.add(f_idx)
		if beta == 0.0:
			beta_pseudo = np.min(self.weights.w)
			if beta_pseudo == 0.0:
				beta_pseudo = 1e-20
			alpha = np.log(1./beta_pseudo)
			print('!!!!!!!!!!!!!! belta is zoro !!!!!!!!!!!!!!', beta_pseudo, np.min(self.weights.w))
		else:
			alpha = np.log(1./beta)

		print('epsilon ', epsilon, 'beta ', beta, 'alpha ', alpha, ' ', len(self.used_feature))
		print('correct = ', np.sum(correct), ' img_idx = ', img_idx)
		
		self.weights.normalize()
		return (f_idx, img_idx, parity, alpha)
		
	def verify(self):
		return
