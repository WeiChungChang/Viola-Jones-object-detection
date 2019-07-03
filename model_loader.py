'''
    File name: utils.py
    Author: WeiChung Chang : r97922153@gmail.com
    Date created: 07/02/2019
    Date last modified:
    Python Version: 3.5
'''


def class ModelLoader():
	def __inint__(self, model_path):
		wcs = np.load('./StrongClassifier.npz')
		csc = CascadedStrongClassifier(wcs)
		
