# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 09:58:48 2021

@author: =GV=
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import math
import matplotlib.pyplot as plt

# 1st model using tensorflow
INITIAL_w = 10.0    # define initial guess
INITIAL_b = 10.0
LEARNING_RATE = 0.09
class Model(object):  # define model (linear regression)
  def __init__(self):
    self.w = tf.Variable(INITIAL_w)     # initialize model values (weight)
    self.b = tf.Variable(INITIAL_b)     # initialize model values (bias)

  def __call__(self, x):
    return self.w*x + self.b

def loss(predicted_ys, target_ys):  # define loss function (~mean_squared_error)
  return tf.reduce_mean(tf.square(predicted_ys - target_ys))

def train(model, xs, ys, learning_rate=LEARNING_RATE):  # define train function/procedure to fit model
  with tf.GradientTape() as gt:   # tracks the value of predicted ys to measure gradient
    current_loss = loss(model(xs), ys)
  dw, db = gt.gradient(current_loss, [model.w, model.b])  # differentiate model values (e.g. weight & bias) with respect to loss function and gradient
  model.w.assign_sub(learning_rate * dw)  # optimizes model values (i.e. weight) based on the learning rate chosen
  model.b.assign_sub(learning_rate * db)  # optimizes model values (i.e. bias) based on the learning rate chosen
  return current_loss

xs = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]    # input/data
ys = [-3.0, -1.0, 1.0, 3.0, 5.0, 7.0]   # output/answers/labels

epochs = range(50)
model = Model()   # instantiate model
list_w, list_b = [], []
losses = []
for epoch in epochs:   #train model n times where n is range and an epoch performs all the steps
  list_w.append(model.w.numpy())
  list_b.append(model.b.numpy())
  current_loss = train(model, xs, ys, learning_rate=0.1)
  losses.append(current_loss)
  print(f'model epoch {epoch + 1}:\nweight = {list_w[-1]}, \tbias = {list_b[-1]}\nloss = {math.sqrt(float(current_loss))}')


# 2nd model using keras
model2 = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])   # defines a neuralnet (Sequential) with 1 layer (Dense), rach layer containing 1 neuron (units) and each neuron evaluates 1 input (input_shape)
model2.compile(optimizer='sgd', loss='mean_squared_error')  # compiles the defined neuralnet into a model with an optimizer ('sgd' = Stochastic Gradient Descent) and a loss function ('mean_squared_error' measures the square root of the mean of sum of the squares of the loss between the predicted ys and target ys math.sqrt(mean(sum([(i[0] - i[-1]) ** 2 for i in zip(predicted_ys, target_ys)]))) used for training/fitting the model )

xs2 = np.array(xs[:], dtype=float)  # input/data ; tensorflow expects data to be a numpy array
ys2 = np.array(ys[:], dtype=float)  # # output/answers/labels ; tensorflow expects data to be a numpy array

model2.fit(xs2, ys2, epochs=200)  # trains/fits model n times where n is epochs

test_x = 10
print(f'\nmodel1 prediction:\nif x = {test_x}, predicts y = {model(test_x)}')
print(f'model2 prediction:\nif x = {test_x}, predicts y = {model2.predict([test_x])[0][0]}')


# plot the model weight and bias adjustments relative to true weight and true bias
TRUE_w = 2.0
TRUE_b = -1.0
plt.plot(epochs, list_w, 'r', epochs, list_b, 'b')
plt.plot([TRUE_w] * len(epochs), 'r--', [TRUE_b] * len(epochs), 'b--')
plt.legend(['w', 'b', 'True w', 'True b'])
plt.show()