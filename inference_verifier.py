'''
    File name: utils.py
    Author: WeiChung Chang
    Date created: 06/29/2019
    Date last modified:
    Python Version: 3.5
'''
import numpy as np

class InferenceVerifier():
	def __init__(self, labels, inf_faces):
		self.labels         = labels
		self.inf_faces      = inf_faces
		if False and len(self.labels.shape) != len(inf_faces.shape): 
			self.labels         = np.reshape(self.labels.shape[0],    1, 1)	
			self.inf_faces      = np.reshape(self.inf_faces.shape[0], 1, 1)		
		
		self.inf_non_faces  = 1 - inf_faces	
		self.neg_labels     = 1 - labels
		print()

	def verify(self, verbose=False):


		true_positive  = np.logical_and(self.inf_faces,     self.labels)
		false_positive = np.logical_xor(self.inf_faces,     true_positive)
		true_negative  = np.logical_and(self.inf_non_faces, self.neg_labels)
		false_negative = np.logical_xor(self.inf_non_faces, true_negative) 

		true_positive_num  = np.sum(true_positive)
		false_positive_num = np.sum(false_positive)
		true_negative_num  = np.sum(true_negative)
		false_negative_num = np.sum(false_negative)
		inf_faces_num      = np.sum(self.inf_faces)
		inf_non_faces_num  = np.sum(self.inf_non_faces)
		faces_num          = np.sum(self.labels)
		non_faces_num      = np.sum(self.neg_labels)

		total              = true_positive_num + false_positive_num + true_negative_num + false_negative_num

		true_positive_rate   = ( true_positive_num  / inf_faces_num     )
		false_positive_rate  = ( false_positive_num / inf_faces_num     )
		if inf_non_faces_num != 0:
			true_negative_rate   = ( true_negative_num  / inf_non_faces_num )
			false_negative_rate  = ( false_negative_num / inf_non_faces_num )
		else:
			true_negative_rate  = 1.0
			false_negative_rate = 1.0
		positive_racall_rate = ( true_positive_num  / faces_num         )
		negative_racall_rate = ( true_negative_num  / non_faces_num     )

		if verbose == True:
			print("======================================================")
			print('\n')
			print( '************ # ************')
			print( 'total #            = %7d' % total             )
			print( '# of faces         = %7d' % faces_num         )
			print( '# of non-faces     = %7d' % non_faces_num     )
			print( '# of inf faces     = %7d' % inf_faces_num     )
			print( '# of inf non-faces = %7d' % inf_non_faces_num )

			print('\n')
			#print( '# of true_positive_num = %7d' % true_positive_num )
				

			print( '\n')
			print( '************ rate ************')
			print( 'positive recall rate = %8f' % positive_racall_rate )
			print( 'negative recall rate = %8f' % negative_racall_rate )
			print( 'true positive rate   = %8f' % true_positive_rate   )
			print( 'false positive rate  = %8f' % false_positive_rate  )
			print( 'true negative rate   = %8f' % true_negative_rate   )
			print( 'false negative rate  = %8f' % false_negative_rate  )
			print( '\n')

			print( '# of false_negative_num = %7d' % false_negative_num )
			print( '# of false_positive_num = %7d' % false_positive_num )

		res = { 
			'true_positive'  : true_positive, 
			'false_positive' : false_positive,
			'true_negative'  : true_negative, 
			'false_negative' : false_negative
			}
		return res
