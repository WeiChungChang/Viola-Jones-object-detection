import numpy as np
import utils
import collections

from tqdm import tqdm
import time

from train_feature_map_generator import TrainFeatureMapGenerator
from weak_classifier_trainer import WeakClassifierTrainer
from weight import Weights

from inference import strong_classifier 

import cv2

def main():
	training_path = '/data/daily/VJ/faces/train'
	test_path     = '/data/daily/VJ/faces/test'

	train_data         = np.arange(3*3).reshape(1,3,3)
	train_data         = np.array([[3,8,5],[2,12,8],[7,1,3]]).reshape(1,3,3)

	sc = strong_classifier(None)

	sc.setImg(train_data)
	#sc.inference()
	r = sc.caculate_expand((2,3),'v',2)
	print('r = \n', r)
if __name__ == '__main__':
	main()
