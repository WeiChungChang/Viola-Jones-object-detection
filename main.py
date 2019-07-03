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

from inference import strong_classifier 

import cv2

def main():
	training_path = '/data/daily/VJ/faces/train'
	test_path     = '/data/daily/VJ/faces/test'

	train_data, labels = utils.load_imgs(training_path)
	train_data         = train_data.astype(np.float32)

	print('train_data ', train_data.shape)

	s = time.time()
	#train_data = utils.normalize_img_cv(train_data)
	#train_data = utils.normalize_img_max(train_data)
	e = time.time()
	print('normalize', e-s)
	#return

	counter=collections.Counter(labels)

	fc = TrainFeatureMapGenerator(train_data)
	fc.set_min_feture(4, 4)
	fc.set_max_feture(4, 4)
	fc.caculate_features_all()
	
	s = time.time()
	fc.prepare_to_train(do_argsort=False)
	e = time.time()
	print('prepare to train', e-s)
	'''
	img_idx = 0
	c1 = fc.fmap[:,img_idx] 
	print('c1.shape ', c1.shape, np.sum(c1), np.sum(np.abs(c1)))

	f_all = []
	for i in tqdm(range(fc.fmap.shape[0])):
		_, fval, _ = fc.get_feature(img_idx, i)
		f_all.append(fval)

	f_all = np.asarray(f_all)
	print(f_all.shape, np.sum(f_all), np.sum(np.abs(f_all)))
	'''
	w  = Weights(labels)
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





	sc = strong_classifier(wc)
	tmp = train_data[0].astype(int)
	tmp = tmp.reshape(19,19,1)
	cv2.imwrite("0.jpg", tmp)
	#train_data = train_data[0:5,:,:]
	sc.setImg(train_data, labels=labels)
	false_samples = sc.inference()







	m  = np.ones(w.w.shape[0])
	fp = np.power(2,false_samples['false_positive'])
	fn = np.power(3,false_samples['false_negative'])

	m  = m * fp
	m  = m * fn
	w2 = Weights2(labels, multiplier=[m])
	np.savetxt('m.txt', m, fmt='%f')
	np.savetxt('w2.txt', w2.w, fmt='%f')

	wt2 = WeakClassifierTrainer(fc.fmap, w2, labels, tmp_val=fc.tmp_val,  argsorted=False)
	wc2 = []
	for i in range(10):
		r = wt2.train()
		uid, fval, coord = fc.get_feature(r[0], r[1])
		wc2.append((uid, fval, coord, r[2], r[3]))
		print('\n\n')
	e = time.time()
	print('train', e-s)

	print(len(wc2))
	for item in wc2:
		print(item)
	print('\n\n')


if __name__ == '__main__':
	main()
