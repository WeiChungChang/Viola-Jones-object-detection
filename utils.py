'''
    File name: utils.py
    Author: WeiChung Chang : r97922153@gmail.com
    Date created: 07/02/2019
    Date last modified:
    Python Version: 3.5
'''

import numpy as np
import os
import cv2

import time
import sys

def check_array_equal (a, b):
	return np.sum(np.abs(a - b))

def __load_imgs_scaled(trafin_folder, h, w, file_ext, neg_folder, max_=sys.float_info.min):
	train_data = []
	labels     = []
	pos_num = 0
	neg_num = 0
	print(trafin_folder)
	for dirpath, dirnames, filenames in os.walk(trafin_folder):
		for filename in [f for f in filenames if f.endswith(file_ext)]:
			img = cv2.imread(dirpath+'/'+filename, cv2.IMREAD_GRAYSCALE)
			img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
			label = 1
			if dirpath.split('/')[-1] == neg_folder:
				label = 0
				neg_num += 1
			else:
				pos_num += 1
			train_data.append(img)
			labels.append(label)
	train_data = np.array(train_data)
	labels     = np.array(labels)
	return train_data, labels

def __load_imgs_native(trafin_folder, file_ext, neg_folder, max_, shuffle):
	train_data = []
	labels     = []
	pos_num = 0
	neg_num = 0

	for dirpath, dirnames, filenames in os.walk(trafin_folder):
		fs = [f for f in filenames if f.endswith(file_ext)]
		total = 0
		'''
		if shuffle == True:
			idx = np.random.permutation(len(fs))
			fs  = fs[idx]
		'''
		#for filename in [f for f in filenames if f.endswith(file_ext)]:
		for filename in fs:
			img = cv2.imread(dirpath+'/'+filename, cv2.IMREAD_GRAYSCALE)
			label = 1
			if dirpath.split('/')[-1] == neg_folder:
				var = np.var(img)
				if var == 0:
					print('@@@@@@@@@@@@ ', filename)
				label = 0
				neg_num += 1
			else:
				pos_num += 1
			train_data.append(img)
			labels.append(label)
			total = total + 1
			if max_ < total:
				break
	train_data = np.array(train_data)
	labels     = np.array(labels)
	'''
	if shuffle == True:
		idx = np.random.permutation(labels.shape[0])
		train_data = train_data[idx]
		labels     = labels[idx]
	'''
	return train_data, labels

def load_imgs(trafin_folder, w=-1, h=-1, file_ext='.pgm', neg_folder='non-face', max_=sys.float_info.max, shuffle=False):
	print('load_imgs ', w, h)
	if w == -1 and h == -1:
		return __load_imgs_native(trafin_folder, file_ext, neg_folder, max_, shuffle)
	else:
		return __load_imgs_scaled(trafin_folder, w, h, file_ext, neg_folder, max_, shuffle)

def pad_zeros_to_imgs(imgs):
	l= len(imgs.shape)
	if l == 3: # "NHW"
		pad_width = ((0,0),(1,0),(1,0))
	elif l == 4: # 'NHWC'
		pad_width = ((0,0),(1,0),(1,0),(0,0))
	else: # 'HW'
		if l != 2:
			print("l = ", l)
			raise RuntimeError('unsupported dim')
		pad_width = ((1,0),(1,0))
		
	return np.pad(array=imgs,pad_width=pad_width,mode='constant',constant_values=0)

def integral_array(imgs):
	l= len(imgs.shape)
	r = np.zeros(imgs.shape)
	if l == 3 or l == 4: # "NHW" or "NHWC"
		r = np.cumsum(imgs, axis=1)
		r = np.cumsum(r,    axis=2)
	else: # 'HW'
		if l != 2:
			raise RuntimeError('unsupported dim')
		r = np.cumsum(imgs, axis=0)
		r = np.cumsum(r,    axis=1)
		
	return r

def integral_array_vertical(imgs):
	l= len(imgs.shape)
	r = np.zeros(imgs.shape)
	if l == 3 or l == 4: # "NHW" or "NHWC"
		r = np.cumsum(imgs,    axis=1)
	else: # 'HW'
		if l != 2:
			raise RuntimeError('unsupported dim')
		r = np.cumsum(imgs,    axis=0)		
	return r	

def integral_array_horizontal(imgs):
	l= len(imgs.shape)
	r = np.zeros(imgs.shape)
	if l == 3 or l == 4: # "NHW" or "NHWC"
		r = np.cumsum(imgs, axis=2)
	else: # 'HW'
		if l != 2:
			raise RuntimeError('unsupported dim')
		r = np.cumsum(imgs, axis=1)		
	return r

def normalize_img_cv(imgs, store=False):
	l= len(imgs.shape)
	print('@@@@@@@@@@ imgs.shape ', imgs.shape)
	print(imgs.shape)
	if store == True:
		for i in range(imgs.shape[0]):
			fname = './check/' + str(i) + '.jpg'
			tmp = imgs[i]
			tmp = tmp.astype(int)


	if l == 3: # "NHW"
		h      = imgs.shape[1]
		w      = imgs.shape[2]
		imgs   = imgs.reshape(imgs.shape[0],-1)
		s      = np.sum(imgs, axis=1)
		area   = imgs.shape[1]
		avg    = s / area
		square = imgs * imgs
		avg_sq = np.sum(square, axis=1) / area
		var    = avg_sq - (avg * avg)
		_t     = np.argwhere(var < 0)
		print('@@@@@@@@@@ var', type(_t), len(_t), _t, var[_t])
		print('@@@@@@@@@@ imgs.shape ', imgs.shape)
		
		
		#masks  = np.argwhere(var==0.0)
		#np.savetxt('masks.txt', masks, fmt='%d')
		
		
		var    = np.array([var,] * imgs.shape[1]).T
		
		#print('@@@@@@@@@@@@ ', var.shape)
		masks  = np.argwhere(var==0.0)
		
		print('@@@@@@@@@@@@@@@@ ', len(masks), var.shape)
		#print('len(idx), ', len(idx), idx)
		#var[idx] = 1.0
		
		
		imgs   = imgs / var
		imgs   = imgs.reshape(imgs.shape[0], h, w)
	else:
		raise RuntimeError('image should be in the format of NHW')

	return imgs, masks

def normalize_img_max(imgs):
	l= len(imgs.shape)
	if l == 3: # "NHW"
		h    = imgs.shape[1]
		w    = imgs.shape[2]
		imgs = imgs.reshape(imgs.shape[0],-1)

		m    = np.max(np.abs(imgs), axis=1)
		#m   = np.max(imgs, axis=1)
		m    += 1e-10
		m    = np.array([m,] * imgs.shape[1]).T 
		imgs = imgs / m
		imgs = imgs.reshape(imgs.shape[0], h, w)
	else:
		raise RuntimeError('image should be in the format of NHW')

	return imgs
