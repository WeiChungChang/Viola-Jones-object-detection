import numpy as np
import collections

class Weights():
	def __init__(self, labels, init_neg_w=1.0, init_pos_w=1.0):
		counter=collections.Counter(labels)

		self.positive_num = counter[1] 
		self.negative_num = counter[0]
		self.total_num    = self.positive_num + self.negative_num
		self.w            = np.zeros(self.total_num)

		s = init_pos_w + init_neg_w
		pos_w = init_pos_w / (s * self.positive_num)
		neg_w = init_neg_w / (s * self.negative_num)			

		self.w = self.w + (labels * pos_w)
		self.w = self.w + ((1 - labels) * neg_w)

		self.normalize()

	def normalize(self):
		self.w = self.w / np.sum(self.w)
		#print('w sum = ', np.sum(self.w))
	

class UnitWeights():

	def __init__(self, labels):
		counter=collections.Counter(labels)

		self.positive_num = counter[0] 
		self.negative_num = counter[1]
		self.total_num    = self.positive_num + self.negative_num
		self.w            = np.ones(self.total_num)

	def normalize(self):
		print("Do nothing")

class Weights2():
	def __init__(self, labels, init_neg_w=1.0, init_pos_w=1.0, multiplier=[]):
		counter=collections.Counter(labels)

		self.positive_num = counter[1] 
		self.negative_num = counter[0]
		self.total_num    = self.positive_num + self.negative_num
		self.w            = np.zeros(self.total_num)

		s = init_pos_w + init_neg_w
		pos_w = init_pos_w / (s * self.positive_num)
		neg_w = init_neg_w / (s * self.negative_num)

		self.w = self.w + (labels * pos_w)
		self.w = self.w + ((1 - labels) * neg_w)

		if len(multiplier) != 0:
			self.w = self.w *  multiplier[0]

		self.normalize()

	def normalize(self):
		self.w = self.w / np.sum(self.w)
		#print('w sum = ', np.sum(self.w))
