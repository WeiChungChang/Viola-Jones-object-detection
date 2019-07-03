'''
    File name: cascade_strong_classifier.py
    Author: WeiChung Chang
    Date created: 06/10/2019
    Date last modified:
    Python Version: 3.5
'''

import numpy as np
#from inference import StrongClassifier 
from strong_classifier import StrongClassifier 

class CascadedStrongClassifier():

	def __init__(self, strong_clfs = None):
		self.strong_clfs = strong_clfs

	def load_model(self, model_path, name='sc'):
		loaded           = np.load(model_path)
		self.strong_clfs = loaded[name]

	def inference(self, img, verbose=False):

		pos = np.ones(img.shape[0])
		if verbose == True:
			detail = np.zeros((len(self.strong_clfs), img.shape[0]))

		for i, wcs in enumerate(self.strong_clfs):
			sc  = StrongClassifier(wcs, detect_window_sz=(24,24))
			sc.setImg(img = img)

			ir  = sc.inference()
			pos = np.logical_and(pos, ir['faces'])
			if verbose == True:
				detail[i] = ir['faces']

		res = {'positive' : pos}
		if verbose == True:
			res['detail'] = detail
		return res
	
	def profile(self):
		return
