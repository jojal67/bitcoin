# -*- coding: utf-8 -*-
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout 
from keras.models import load_model, model_from_json
from keras.utils import to_categorical
from keras.utils import np_utils



a=np.array([ 0, 1,  0,  1])
a = to_categorical(a,num_classes=2)

print(a)
