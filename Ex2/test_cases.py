# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 16:28:21 2022

@author: niky_
"""

from costFunction import *

# testing sigmoid function implementation
assert sigmoid(0)==0.5
assert (sigmoid([0,1,1]) == np.array([sigmoid(0), sigmoid(1), sigmoid(1)])).all()



print("All tests passed!")