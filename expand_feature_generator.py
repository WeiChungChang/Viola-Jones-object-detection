'''
    File name: pattern	_factory.py
    Author: WeiChung Chang
    Date created: 05/09/2019
    Date last modified:
    Python Version: 3.5
'''

import numpy as np
from enum import Enum

def vertical2_expand_features_generator(pad_img_shape):
	ph = pad_img_shape[0]
	pw = pad_img_shape[1]

	h_dim_hf = np.arange(((ph-1)//2)+1)[1:] #[1:] to drop zero
	h_dim    = h_dim_hf * 2
	w_dim    = np.arange(pw)[1:]
	pattern  = np.transpose([np.tile(h_dim, len(w_dim)), np.repeat(w_dim, len(h_dim))])
	return pattern

def vertical3_expand_features_generator(pad_img_shape):
	ph = pad_img_shape[0]
	pw = pad_img_shape[1]

	h_dim_td = np.arange(((ph-1)//3)+1)[1:] #[1:] to drop zero
	h_dim    = h_dim_td * 3
	w_dim    = np.arange(pw)[1:]
	pattern  = np.transpose([np.tile(h_dim, len(w_dim)), np.repeat(w_dim, len(h_dim))])
	return pattern

def horizontal2_expand_features_generator(pad_img_shape):
	ph = pad_img_shape[0]
	pw = pad_img_shape[1]

	w_dim_hf = np.arange(((pw-1)//2)+1)[1:] #[1:] to drop zero
	w_dim    = w_dim_hf * 2
	h_dim    = np.arange(ph)[1:]
	pattern  = np.transpose([np.tile(h_dim, len(w_dim)), np.repeat(w_dim, len(h_dim))])
	return pattern

def horizontal3_expand_features_generator(pad_img_shape):
	ph = pad_img_shape[0]
	pw = pad_img_shape[1]

	w_dim_td = np.arange(((pw-1)//3)+1)[1:] #[1:] to drop zero
	w_dim    = w_dim_td * 3
	h_dim    = np.arange(ph)[1:]
	pattern  = np.transpose([np.tile(h_dim, len(w_dim)), np.repeat(w_dim, len(h_dim))])
	return pattern

def diagonal_expand_features_generator(pad_img_shape):
	ph = pad_img_shape[0]
	pw = pad_img_shape[1]

	h_dim_hf = np.arange(((ph-1)//2)+1)[1:] #[1:] to drop zero
	w_dim_hf = np.arange(((pw-1)//2)+1)[1:] #[1:] to drop zero
	w_dim    = w_dim_hf * 2
	h_dim    = h_dim_hf * 2
	pattern  = np.transpose([np.tile(h_dim, len(w_dim)), np.repeat(w_dim, len(h_dim))])
	return pattern

def generate_expand_feature(type_, pad_img_shape):
    if type_ == 'diagonal':
        type_ = 'horizontal2'
    generator_name = type_ + '_expand_features_generator'

    if generator_name in globals():
        #print(generator_name, globals()[generator_name])
        return globals()[generator_name](pad_img_shape)

    raise RuntimeError('unknown feature type "%s"' % ext)



