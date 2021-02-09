# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 13:33:28 2021

@author: =GV=
"""
import tensorflow as tf
from tensorflow import keras

data = tf.keras.datasets.mnist
(training_images, training_labels), (eval_images, eval_labels) = data.load_data()   # load data sets
training_images = training_images / 255.0   # normalize the training data to be a value between 1 and 0
eval_images = eval_images / 255.0           # normalize the evaluation data to be a value between 1 and 0

layer1 = keras.layers.Dense(units=20, activation=tf.nn.relu)      # define 1st layer with 20 neurons and change any output <0 to 0 (Rectified Linear Model)
layer2 = keras.layers.Dense(units=10, activation=tf.nn.softmax)   # define 2nd layer with 10 neurons and to output only the neuron with the highest value
model = keras.Sequential([keras.layers.Flatten(input_shape=(28,28)), layer1, layer2])   # define a Sequential neuralnet that accepts 28x28 images as input and Flattens input into a 1D image before feeding it to layer1
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])   # compiles model to  use 'adam' optimizer to varie learning_rates to maximize multiple neurons to converge more quickly, loss function 'sparse_categorical_crossnetropy' measures loss across categories (ideal for classification models), and to use 'accuracy' as a metric instead of loss
model.fit(training_images, training_labels, validation_data=(eval_images, eval_labels), epochs=20)    # trains the model while evaluating its accuracy in every epoch


# test model
classifications = model.predict(eval_images)
print(classifications[0])
print(eval_labels[0])