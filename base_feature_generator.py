'''
    File name: base_feature_generator.py
    Author: WeiChung Chang : r97922153@gmail.com
    Date created: 07/02/2019
    Date last modified:
    Python Version: 3.5
'''

import numpy as np
from enum import Enum

def vertical2_base_feature_generator(raw_img_shape):
	h = raw_img_shape[0]

	h_dim_hf = np.arange(((h)//2)+1)[1:] #[1:] to drop zero
	h_dim    = h_dim_hf * 2
	w_dim    = np.array([1])
	pattern  = np.transpose([np.tile(h_dim, len(w_dim)), np.repeat(w_dim, len(h_dim))])
	return pattern

def vertical3_base_feature_generator(raw_img_shape):
	h = raw_img_shape[0]

	h_dim_td = np.arange(((h)//3)+1)[1:] #[1:] to drop zero
	h_dim    = h_dim_td * 3
	w_dim    = np.array([1])
	pattern  = np.transpose([np.tile(h_dim, len(w_dim)), np.repeat(w_dim, len(h_dim))])
	return pattern

def horizontal2_base_feature_generator(raw_img_shape):
	w = raw_img_shape[1]

	w_dim_hf = np.arange(((w)//2)+1)[1:] #[1:] to drop zero
	w_dim    = w_dim_hf * 2
	h_dim    = np.array([1])
	pattern  = np.transpose([np.tile(h_dim, len(w_dim)), np.repeat(w_dim, len(h_dim))])
	return pattern

def horizontal3_base_feature_generator(raw_img_shape):
	w = raw_img_shape[1]

	w_dim_td = np.arange(((w)//3)+1)[1:] #[1:] to drop zero
	w_dim    = w_dim_td * 3
	h_dim    = np.array([1])
	pattern  = np.transpose([np.tile(h_dim, len(w_dim)), np.repeat(w_dim, len(h_dim))])
	return pattern

def diagonal_base_feature_generator(raw_img_shape):
	w = raw_img_shape[1]

	w_dim_hf = np.arange(((w)//2)+1)[1:] #[1:] to drop zero
	w_dim    = w_dim_hf * 2
	h_dim    = np.array([1])
	pattern  = np.transpose([np.tile(h_dim, len(w_dim)), np.repeat(w_dim, len(h_dim))])
	return pattern

def generate_base_feature (type_, raw_img_shape):
    generator_name = type_ + '_base_feature_generator'
    if generator_name in globals():
        #print(generator_name, globals()[generator_name])
        return globals()[generator_name](raw_img_shape)

    raise RuntimeError('unknown feature type "%s"' % ext)



