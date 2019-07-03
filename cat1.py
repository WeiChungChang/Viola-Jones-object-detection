'''
    File name: cat.py
    Author: WeiChung Chang : r97922153@gmail.com
    Date created: 07/02/2019
    Date last modified:
    Python Version: 3.5
'''

import numpy as np
import utils
import collections

from tqdm import tqdm
import time

from train_feature_map_generator import TrainFeatureMapGenerator
from weak_classifier_trainer import WeakClassifierTrainer

from weight import Weights
from weight import UnitWeights
from weight import Weights2

from inference import StrongClassifier
#from strong_classifier import StrongClassifier 

from inference_verifier import InferenceVerifier 

from cascaded_strong_classifier import CascadedStrongClassifier

import cv2

def main():
	training_path = '/data/cat/train'
	test_path     = '/data/cat/test'

	train_data, labels = utils.load_imgs(training_path, file_ext='.jpg', neg_folder='non-cat')
	train_data         = train_data.astype(np.float32)

	s = time.time()
	train_data, masks = utils.normalize_img_cv(train_data)
	#train_data = utils.normalize_img_max(train_data)
	e = time.time()
	print('\n\n normalize\n\n', e - s)
	print('train_data ', train_data.shape)

	s = time.time()
	#train_data = utils.normalize_img_cv(train_data)
	#train_data = utils.normalize_img_max(train_data)
	e = time.time()
	print('normalize', e-s)
	#return

	counter=collections.Counter(labels)

	fc = TrainFeatureMapGenerator(train_data)
	fc.set_min_feture(2, 2)
	fc.set_max_feture(2, 2)
	fc.caculate_features_all()
	
	s = time.time()
	fc.prepare_to_train(do_argsort=False)
	e = time.time()
	print('prepare to train', e-s)

	print(fc.fmap.shape)

	w  = Weights(labels, init_neg_w=1.0, init_pos_w=1.19)
	#w  = Weights(labels)
	print(w.positive_num, w.negative_num)
	#uw = UnitWeights(labels)
	wt = WeakClassifierTrainer(fc.fmap, w, labels, tmp_val=fc.tmp_val,  argsorted=False)
	#wt = WeakClassifierTrainer(fc.fmap, uw, labels, tmp_val=fc.tmp_val,  argsorted=False)


	wc = []
	s = time.time()
	for i in range(1):
		r = wt.train()
		uid, fval, coord = fc.get_feature(r[0], r[1])
		wc.append((uid, fval, coord, r[2], r[3]))
		print('\n\n')
	e = time.time()
	print('train', e-s)

	print(len(wc))
	for item in wc:
		print(item)
	print('\n\n')






	sc = StrongClassifier(wc)
	sc.setImg(train_data, labels=labels)
	ir = sc.inference()

	iv = InferenceVerifier(labels, ir['faces'])
	vr = iv.verify(verbose=True)


if __name__ == '__main__':
	main()
