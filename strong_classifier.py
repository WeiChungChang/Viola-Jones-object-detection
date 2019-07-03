'''
    File name: utils.py
    Author: WeiChung Chang
    Date created: 06/26/2019
    Date last modified:
    Python Version: 3.5
'''
import numpy as np

from expand_feature_generator import generate_expand_feature

from utils import integral_array
from utils import pad_zeros_to_imgs

SIGN = 0
Y_S  = 1
Y_E  = 2
X_S  = 3
X_E  = 4

class StrongClassifier():
	"""
	Strong Classifier

	Attributes
	----------
	weak_classifiers : class WeakClassifier
		image set to train
	ingegral_h : ndarray
		integral img at horizotal direction
	ingegral_v : ndarray
		integral img at vertical direction
	self.h : int
		raw image height
	self.w : int
		raw image width

	Methods
	-------
	says(sound=None)
		Prints the animals name and what sound it makes
	"""
	def __init__(self, weak_classifiers, detect_window_sz):

		self.weak_classifiers = weak_classifiers
		self.detect_window_sz = detect_window_sz
		self.integral_img     = None
		self.labels           = None



		self.dispatch = {
			('v', 2) : self.__vertical2_expand_to_coordinate,
			('h', 2) : self.__horizontal2_expand_to_coordinate,
			('v', 3) : self.__vertical3_expand_to_coordinate,
			('h', 3) : self.__horizontal3_expand_to_coordinate,
		}

	def setImg(self, img, labels=[], integral=False, backup=False):
		if integral == False:
			if backup == True:
				self.img = img.copy()
			else:
				self.img = None
          	
			self.integral_img = pad_zeros_to_imgs(img) # (N, H, W)
			self.integral_img = integral_array(self.integral_img)
			self.h            = self.integral_img.shape[1]
			self.w            = self.integral_img.shape[2]
			print(self.h, self.w)
			print('self.integral_img = \n', self.integral_img.shape)
			if labels != []:
				self.labels = labels
			else:
				self.labels = []
		else: # TODO
			raise RuntimeError('TODO: handle non-integral input')
	
	def inference(self, masks=[], store_score=False, verbose=False):
		d_y = self.h - self.detect_window_sz[0]
		d_x = self.w - self.detect_window_sz[1]

		#score           = np.zeros(self.integral_img.shape[0])
		score            = np.zeros((self.integral_img.shape[0], d_y, d_x))
		face_threshold  = 0.0

		if store_score == True:
			scores = np.zeros((len(self.weak_classifiers), self.integral_img.shape[0]))

		for i, item in enumerate(self.weak_classifiers):
			dir_                 = item[0][0]
			pattern_major_length = item[0][1]			
			expand               = item[0][2]
			threshold            = item[1]
			pos_y                = item[2][0]
			pos_x                = item[2][1]
			parity               = item[3]
			alpha                = item[4]
			print('item ', item, dir_, pattern_major_length, expand)


			face_threshold += alpha

			if verbose == True:
				print('expand               = ', expand)
				print('dir_                 = ', dir_)
				print('pattern_major_length = ', pattern_major_length)
				print('parity               = ', parity)
				print('threshold            = ', threshold)
				print('pos_y                = ', pos_y)
				print('pos_x                = ', pos_x)

			r = self.caculate_expand(expand, dir_, pattern_major_length)

			'''
			print('@@@@@@@@@@@@ ', r.shape, pos_y, pos_x)
			d_y = self.detect_window_sz[0] - pos_y
			d_x = self.detect_window_sz[1] - pos_x
			print('@@@@@@@@@@@@ ', pos_y, (r.shape[1] + 1 - d_y + 1), pos_x, (r.shape[2] + 1 - d_x + 1))
			r = r[:, pos_y:(r.shape[1] + 1 - d_y + 1), pos_x:(r.shape[2] + 1 - d_x + 1)]
			#r = np.squeeze(r)
			print('****** ', r.shape, r, np.sum(r))
			'''


			print('@@@@ d_y , d_x ', d_y, d_x)
			r = r[:, pos_y:(pos_y + d_y), pos_x:(pos_x + d_x)]
			#r = np.squeeze(r)

			if parity == -1:
				invert = r >= threshold
				judge  = r <  threshold
			else:
				invert = r <= threshold
				judge  = r > threshold
				
			judge = judge.astype(int)
 
			score = score + (judge * alpha)

			if store_score == True:
				scores[i] = score

		if False and store_score == True:
			np.savetxt('faces.txt',     scores[:,0:102], fmt='%4.7f')
			np.savetxt('non-faces.txt', scores[:,101:],  fmt='%4.7f')

		print('@@@@@@@@@ score.shape ', score.shape, np.sum(score))
		#score = np.squeeze(score)

		face_threshold = face_threshold * 0.41
		print('face_threshold = ', face_threshold)
		faces      = (score >= face_threshold).astype(int)
		non_faces  = 1 - faces

		print('@@@@@@@@@ faces.shape ', faces.shape, np.sum(faces), non_faces.shape, np.sum(non_faces))
	
		return { 'faces' : faces, 'non_faces' : non_faces}

	def inference_window (self, window):
		return


	def caculate_expand (self, expand, dir_, pattern_major_length):
		if dir_ == 'd':
			p, r = self.dispatch[('h', 2)]( (expand[0]//2, expand[1]) )
			#p, r = self.dispatch[('h', 2)](expand)
		else:
			p, r = self.dispatch[(dir_, pattern_major_length)](expand)

		res  = self.__caculate_expand (p, r)
		if dir_ == 'd':
			res = self.__calculate_diagonal(res, expand)
			return res
		return res

	def __caculate_expand (self, coord, range_):
		r = np.zeros((self.integral_img.shape[0], range_[0], range_[1]))
		for t in coord:
			tmp = (t[SIGN] * self.integral_img[:,t[Y_S]:t[Y_E],t[X_S]:t[X_E]])
			r += tmp
		return r



	def __calculate_diagonal(self, hfmap, expand):
		fh = expand[0] // 2
		fw = expand[1]
		d  = hfmap.shape[1] - fh
		fdig = []
		if (d > 0):
			fdig = (hfmap[:,0:d,:] - hfmap[:,fh:,:])
		return fdig

	def __vertical2_expand_to_coordinate(self, expand):

		p_h = expand[0]
		p_w = expand[1]

		r_h = self.h - p_h
		r_w = self.w - p_w

		p_rb = ( 1, p_h,    self.h,         p_w, self.w)
		p_rt = (-2, p_h//2, ((p_h//2)+r_h), p_w, self.w)
		p_lb = (-1, p_h,    self.h,         0,   r_w   )
		p_lt = ( 2, p_h//2, ((p_h//2)+r_h), 0,   r_w   )

		n_rt = ( 1, 0, r_h, p_w, self.w  )
		n_lt = (-1, 0, r_h, 0,   r_w )

		return [p_rb, p_rt, p_lb, p_lt, n_rt, n_lt], (r_h, r_w)

	def __horizontal2_expand_to_coordinate(self, expand):

		p_h = expand[0]
		p_w = expand[1]

		r_h = self.h - p_h
		r_w = self.w - p_w

		p_rb = ( 1, p_h, self.h, p_w,    self.w        )
		p_rt = (-1, 0,   r_h,    p_w,    self.w        )
		p_lb = (-2, p_h, self.h, p_w//2, ((p_w//2)+r_w))
		p_lt = ( 2, 0,   r_h,    p_w//2, ((p_w//2)+r_w))

		n_lb = (-1,  0,   r_h,    0, r_w)
		n_lt = ( 1,  p_h, self.h, 0, r_w)

		return [p_rb, p_rt, p_lb, p_lt, n_lb, n_lt], (r_h, r_w)

	def __vertical3_expand_to_coordinate(self, expand):

		p_h = expand[0]
		p_w = expand[1]

		r_h = self.h - p_h
		r_w = self.w - p_w

		p_rb_b = ( 1, p_h,        self.h,           p_w, self.w)
		p_rt_b = (-2, (p_h//3)*2, ((p_h//3)*2+r_h), p_w, self.w)
		p_lb_b = (-1, p_h,        self.h,           0,   r_w   )
		p_lt_b = ( 2, (p_h//3)*2, ((p_h//3)*2+r_h), 0,   r_w   )

		n_rt = ( 2, (p_h//3), ((p_h//3)+r_h), p_w, self.w)
		n_lt = (-2, (p_h//3), ((p_h//3)+r_h), 0,   r_w   )

		p_rt_t = (-1, 0, r_h, p_w, self.w)
		p_lt_t = ( 1, 0, r_h, 0,   r_w   )

		return [p_rb_b, p_rt_b, p_lb_b, p_lt_b, n_rt, n_lt, p_rt_t, p_lt_t], (r_h, r_w)
	
	def __horizontal3_expand_to_coordinate(self, expand):

		p_h = expand[0]
		p_w = expand[1]

		r_h = self.h - p_h
		r_w = self.w - p_w

		p_rb_r = ( 1, p_h, self.h, p_w,        self.w          )
		p_rt_r = (-1, 0,   r_h,    p_w,        self.w          )
		p_lb_r = (-2, p_h, self.h, (p_w//3)*2, ((p_w//3)*2+r_w))
		p_lt_r = ( 2, 0,   r_h,    (p_w//3)*2, ((p_w//3)*2+r_w))

		n_lb = ( 2, p_h, self.h, (p_w//3), ((p_w//3)+r_w))
		n_lt = (-2, 0,   r_h,    (p_w//3), ((p_w//3)+r_w))

		p_lt_l = ( 1, 0,   r_h,    0, r_w)
		p_lb_l = (-1, p_h, self.h, 0, r_w)

		return [p_rb_r, p_rt_r, p_lb_r, p_lt_r, n_lb, n_lt, p_lt_l, p_lb_l], (r_h, r_w)
