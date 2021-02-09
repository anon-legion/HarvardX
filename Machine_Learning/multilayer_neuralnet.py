# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 10:35:29 2021

@author: =GV=
"""
from tensorflow import keras
import numpy as np

layer1 = keras.layers.Dense(units=2, input_shape=[1])
layer2 = keras.layers.Dense(units=1)
model = keras.Sequential([layer1, layer2])
model.compile(optimizer='sgd', loss='mean_squared_error')
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
model.fit(xs, ys, epochs=500)

test_x = 10
print(f'model prediction:\nif x = {test_x},\tpredicts y = {model.predict([test_x])}')