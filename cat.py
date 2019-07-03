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

#from inference import StrongClassifier
from strong_classifier import StrongClassifier 

from inference_verifier import InferenceVerifier 

from cascaded_strong_classifier import CascadedStrongClassifier

import cv2

def main():
	training_path = './train'
	test_path     = './test'

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
	for i in range(20):
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






	sc = StrongClassifier(wc, (24,24))
	sc.setImg(train_data, labels=labels)
	ir = sc.inference()

	iv = InferenceVerifier(labels, ir['faces'])
	#print('@@@@@@@@@@@@@@@@@@@@ iv ',  ir['faces'].shape)
	vr = iv.verify(verbose=True)



	test_data, test_labels = utils.load_imgs(test_path, file_ext='.jpg', neg_folder='non-cat', max_=3000)
	test_data              = test_data.astype(np.float32)
	
	test_data, masks = utils.normalize_img_cv(test_data, store=True)


	#------------------------------------------------------------------------------------
	m  = np.ones(w.w.shape[0])
	fp = np.power(1.0, vr['false_positive'])
	fn = np.power(1.5, vr['false_negative'])

	m  = m * fp
	m  = m * fn
	w2 = Weights2(labels, init_neg_w=1.0, init_pos_w=3.0, multiplier=[m])
	print("\n\n\n======================================================================\n\n\n")


	fc.clean()
	fc.set_min_feture(4, 4)
	fc.set_max_feture(6, 6)
	
	fc.caculate_features_all()
	fc.prepare_to_train(do_argsort=False)


	wt2 = WeakClassifierTrainer(fc.fmap, w2, labels, tmp_val=fc.tmp_val,  argsorted=False)
	wc2 = []
	s = time.time()
	for i in range(20):
		r = wt2.train()
		uid, fval, coord = fc.get_feature(r[0], r[1])
		wc2.append((uid, fval, coord, r[2], r[3]))
		print('\n\n')
	e = time.time()
	print('train', e-s)










	m  = np.ones(w.w.shape[0])
	fp = np.power(1.0, vr['false_positive'])
	fn = np.power(1.5, vr['false_negative'])

	m  = m * fp
	m  = m * fn
	w3 = Weights2(labels, init_neg_w=1.0, init_pos_w=3.0, multiplier=[m])
	#print("\n\n\n======================================================================\n\n\n")


	fc.clean()
	fc.set_min_feture(8, 8)
	fc.set_max_feture(8, 8)
	
	fc.caculate_features_all()
	fc.prepare_to_train(do_argsort=False)


	wt3 = WeakClassifierTrainer(fc.fmap, w3, labels, tmp_val=fc.tmp_val,  argsorted=False)
	wc3 = []
	s = time.time()
	for i in range(20):
		r = wt3.train()
		uid, fval, coord = fc.get_feature(r[0], r[1])
		wc3.append((uid, fval, coord, r[2], r[3]))
		print('\n\n')
	e = time.time()
	print('train', e-s)



	m  = np.ones(w.w.shape[0])
	fp = np.power(1.0, vr['false_positive'])
	fn = np.power(1.5, vr['false_negative'])

	m  = m * fp
	m  = m * fn
	w4 = Weights2(labels, init_neg_w=1.0, init_pos_w=3.0, multiplier=[m])
	#print("\n\n\n======================================================================\n\n\n")


	fc.clean()
	fc.set_min_feture(10, 10)
	fc.set_max_feture(10, 10)
	
	fc.caculate_features_all()
	fc.prepare_to_train(do_argsort=False)


	wt4 = WeakClassifierTrainer(fc.fmap, w4, labels, tmp_val=fc.tmp_val,  argsorted=False)
	wc4 = []
	s = time.time()
	for i in range(20):
		r = wt4.train()
		uid, fval, coord = fc.get_feature(r[0], r[1])
		wc4.append((uid, fval, coord, r[2], r[3]))
		print('\n\n')
	e = time.time()
	print('train', e-s)




	m  = np.ones(w.w.shape[0])
	fp = np.power(1.0, vr['false_positive'])
	fn = np.power(1.5, vr['false_negative'])

	m  = m * fp
	m  = m * fn
	w5 = Weights2(labels, init_neg_w=1.0, init_pos_w=3.0, multiplier=[m])
	#print("\n\n\n======================================================================\n\n\n")


	fc.clean()
	fc.set_min_feture(12, 12)
	fc.set_max_feture(12, 12)
	
	fc.caculate_features_all()
	fc.prepare_to_train(do_argsort=False)


	wt5 = WeakClassifierTrainer(fc.fmap, w5, labels, tmp_val=fc.tmp_val,  argsorted=False)
	wc5 = []
	s = time.time()
	for i in range(20):
		r = wt5.train()
		uid, fval, coord = fc.get_feature(r[0], r[1])
		wc5.append((uid, fval, coord, r[2], r[3]))
		print('\n\n')
	e = time.time()
	print('train', e-s)







	wcs = [wc, wc2, wc3, wc4, wc5]
	np.savez_compressed('./StrongClassifier', sc=wcs)
	#csc = CascadedStrongClassifier(strong_clfs=wcs)
	csc = CascadedStrongClassifier()
	csc.load_model('./StrongClassifier.npz')
	
	cat_img = cv2.imread('/data/daily/VJ/VJ_git/new_ranged_table/0625/VJ_git/checkin/cat.jpg')
	cat_img = cv2.cvtColor(cat_img, cv2.COLOR_BGR2GRAY)
	cat_img = cat_img.reshape(1, cat_img.shape[0], cat_img.shape[1]) 
	print('cat_img', cat_img.shape)

	print("======================================================================")
	r = csc.inference(cat_img)
	print("======================================================================")

	if False:
		r = csc.inference(test_data)
		res = r['positive']
		print('^^^^', type(res))
		iv2 = InferenceVerifier(test_labels, res)
		vr2 = iv2.verify(verbose=True)
		print("\n\n\n======================================================================\n\n\n")
	#------------------------------------------------------------------------------------



	'''
	print('test_data.shape = ', test_data.shape)
	print(np.sum(test_labels))

	sc = StrongClassifier(wc)
	sc.setImg(test_data, labels=test_labels)
	ir = sc.inference(masks=[masks], store_score=False)
	iv = InferenceVerifier(test_labels, ir['faces'])
	vr = iv.verify(verbose=True)

	np.savetxt('labels.txt', test_labels, fmt='%d')
	'''

if __name__ == '__main__':
	main()
