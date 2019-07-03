'''
    File name: definition.py (r97922153@gmail.com)
    Author: WeiChung Chang
    Date created: 07/02/2019
    Date last modified:
    Python Version: 3.5
'''

from enum import Enum

class patternType(Enum):
    horizontal2   = 'horizontal2'
    horizontal3   = 'horizontal3'
    vertical2     = 'vertical2'
    vertical3     = 'vertical3'
    diagonal      = 'diagonal'

patten_param_table = {
	'horizontal2' : ('h', 2),
	'horizontal3' : ('h', 3),
	'vertical2'   : ('v', 2),
	'vertical3'   : ('v', 3),
	'diagonal'    : ('d', 2), # 2 comes from h
}
