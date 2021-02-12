# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:59:51 2021

@author: =GV=
"""
import tensorflow as tf
from tensorflow import keras

data = tf.keras.datasets.fashion_mnist
(training_data, training_labels), (eval_data, eval_labels) = data.load_data()
training_images = training_data / 255.0
eval_images = eval_data / 255.0

training_epochs = 20

# 1st model
m1_layer1 = keras.layers.Dense(units=20, activation=tf.nn.relu)
m1_layer2 = keras.layers.Dense(units=10, activation=tf.nn.softmax)
model_1 = keras.Sequential([keras.layers.Flatten(input_shape=(28,28)), m1_layer1, m1_layer2])
model_1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print('Model 1:\n')
model_1.fit(training_images, training_labels, validation_data=(eval_images, eval_labels), epochs=training_epochs)

target_accuracy = model_1.evaluate(eval_images, eval_labels)[-1]
print(f'\n Model 1 accuracy = {target_accuracy}')

# 2nd model (Deep Neuralnet)
m2_layer1 = keras.layers.Dense(units=28, activation=tf.nn.relu)
m2_layer2 = keras.layers.Dense(units=512, activation=tf.nn.relu)    # hidden layer
m2_layer3 = keras.layers.Dense(units=10, activation=tf.nn.softmax)
model_2 = keras.Sequential([keras.layers.Flatten(input_shape=(28,28)), m2_layer1, m2_layer2, m2_layer3])
model_2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, log={}):
    if log.get('accuracy') > target_accuracy:
      self.model.stop_training = True

callback = myCallback()

print('\nModel 2:\n')
model_2.fit(training_images, training_labels, validation_data=(eval_images, eval_labels), callbacks=[callback], epochs=training_epochs)

