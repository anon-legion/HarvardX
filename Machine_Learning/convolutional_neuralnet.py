# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 21:23:10 2021

@author: Anon
"""

# comparison of the effects of convolutions and pooling on model accuracy

import tensorflow as tf
from tensorflow import keras

data = tf.keras.datasets.cifar10
(training_data, training_labels), (val_data, val_labels) = data.load_data()
training_images = training_data / 255.0
val_images = val_data / 255.0

num_epochs = 5

m1_layer1 = keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu, input_shape=(32,32,3))
m1_layer2 = keras.layers.MaxPooling2D(2,2)
m1_layer3 = keras.layers.Dense(units=1024, activation=tf.nn.relu)
m1_layer4 = keras.layers.Dense(units=10, activation=tf.nn.softmax)
model_1 = keras.Sequential([m1_layer1, m1_layer2, keras.layers.Flatten(), m1_layer3, m1_layer4])
model_1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
train = model_1.fit(training_images, training_labels, validation_data=(val_images, val_labels), epochs=num_epochs)

training_accuracy1 = train.history['accuracy'][-1]
target_accuracy1 = train.history['val_accuracy'][-1]
print(f'model_1 training accuracy = {training_accuracy1}, \tevaluation accuracy = {target_accuracy1}')


# reshaping data to enhance data normalization vs previous model
data2 = tf.keras.datasets.cifar10
(training_data, training_labels), (val_data, val_labels) = data2.load_data()
training_images2 = (training_data.reshape(len(training_data), 32, 32, 3)) / 255.0
val_images2 = (val_data.reshape(len(val_data), 32, 32, 3)) / 255.0

class myCallback(tf.keras.callbacks.Callback):
  def __init__(self, other):
    self.other = other
  def on_epoch_end(self, epoch, logs={}):
    if logs.get('accuracy') > self.other.history['accuracy'][-1] and logs.get('val_accuracy') > self.other.history['val_accuracy'][-1]:
      self.model.stop_training = True

m2_callback = myCallback(train)

m2_layer1 = keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu, input_shape=(32,32,3))
m2_layer2 = keras.layers.MaxPooling2D(3,3)
m2_layer3 = keras.layers.Dense(units=1024, activation=tf.nn.relu)
m2_layer4 = keras.layers.Dense(units=10, activation=tf.nn.softmax)
model_2 = keras.Sequential([m2_layer1, m2_layer2, keras.layers.Flatten(), m2_layer3, m2_layer4])
model_2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
train2 = model_2.fit(training_images2, training_labels, validation_data=(val_images2, val_labels), epochs=num_epochs, callbacks=[m2_callback])

training_accuracy2 = train2.history['accuracy'][-1]
val_accuracy2 = train2.history['val_accuracy'][-1]
print(f'model_2 training accuracy = {training_accuracy2}, \tevaluation accuracy = {val_accuracy2}')
# reshaping dataset marginally improved validation accuracy of model vs previous model


# add extra identical convolution and pooling layer vs previous model
data3 = tf.keras.datasets.cifar10
(training_data, training_labels), (val_data, val_labels) = data3.load_data()
training_images3 = (training_data.reshape(len(training_data), 32, 32, 3)) / 255.0
val_images3 = (val_data.reshape(len(val_data), 32, 32, 3)) / 255.0

m3_callback = myCallback(train2)

m3_layer1 = keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu, input_shape=(32,32,3))
m3_layer2 = keras.layers.MaxPooling2D(2,2)
m3_layer3 = keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu)
m3_layer4 = keras.layers.MaxPooling2D(2,2)
m3_layer5 = keras.layers.Dense(units=1024, activation=tf.nn.relu)
m3_layer6 = keras.layers.Dense(units=10, activation=tf.nn.softmax)
model_3 = keras.Sequential([m3_layer1, m3_layer2, m3_layer3, m3_layer4, keras.layers.Flatten(), m3_layer5, m3_layer6])
model_3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
train3 = model_3.fit(training_images3, training_labels, validation_data=(val_images3, val_labels), callbacks=[m3_callback], epochs=num_epochs)

training_accuracy3 = train3.history['accuracy'][-1]
val_accuracy3 = train3.history['val_accuracy'][-1]
print(f'model_3 training accuracy = {training_accuracy3}, \tevaluation accuracy = {val_accuracy3}')
# additional convolution layer and pooling layer improved overall validation accuracy whith no significant training accuracy delta vs previous model


# double the convolutions/filters of input layer vs previous model
data4 = tf.keras.datasets.cifar10
(training_data, training_labels), (val_data, val_labels) = data4.load_data()
training_images4 = (training_data.reshape(len(training_data), 32, 32, 3)) / 255.0
val_images4 = (val_data.reshape(len(val_data), 32, 32, 3)) / 255.0

m4_callback = myCallback(train3)

m4_layer1 = keras.layers.Conv2D(128, (3,3), activation=tf.nn.relu, input_shape=(32,32,3))
m4_layer2 = keras.layers.MaxPooling2D(2,2)
m4_layer3 = keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu)
m4_layer4 = keras.layers.MaxPooling2D(2,2)
m4_layer5 = keras.layers.Dense(units=1024, activation=tf.nn.relu)
m4_layer6 = keras.layers.Dense(units=10, activation=tf.nn.softmax)
model_4 = keras.Sequential([m4_layer1, m4_layer2, m4_layer3, m4_layer4, keras.layers.Flatten(), m4_layer5, m4_layer6])
model_4.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
train4 = model_4.fit(training_images4, training_labels, validation_data=(val_images4, val_labels), callbacks=[m4_callback], epochs=num_epochs)

training_accuracy4 = train4.history['accuracy'][-1]
val_accuracy4 = train4.history['val_accuracy'][-1]
print(f'model_4 training accuracy = {training_accuracy4}, \tevaluation accuracy = {val_accuracy4}')
# doubling convolutions/filters of input layer significantly decreased the training accuracy and moderately decreased the overall validation accuracy (~3%) vs previous model


# 2nd
# return convolutions/filters of input layer to original value while doubling convolutions/filters of hidden layer vs previous model
data5 = tf.keras.datasets.cifar10
(training_data, training_labels), (val_data, val_labels) = data5.load_data()
training_images5 = (training_data.reshape(len(training_data), 32, 32, 3)) / 255.0
val_images5 = (val_data.reshape(len(val_data), 32, 32, 3)) / 255.0

m5_callback = myCallback(train4)

m5_layer1 = keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu, input_shape=(32,32,3))
m5_layer2 = keras.layers.MaxPooling2D(2,2)
m5_layer3 = keras.layers.Conv2D(128, (3,3), activation=tf.nn.relu)
m5_layer4 = keras.layers.MaxPooling2D(2,2)
m5_layer5 = keras.layers.Dense(units=1024, activation=tf.nn.relu)
m5_layer6 = keras.layers.Dense(units=10, activation=tf.nn.softmax)
model_5 = keras.Sequential([m5_layer1, m5_layer2, m5_layer3, m5_layer4, keras.layers.Flatten(), m5_layer5, m5_layer6])
model_5.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
train5 = model_5.fit(training_images5, training_labels, validation_data=(val_images5, val_labels), epochs=num_epochs, callbacks=[m5_callback])

training_accuracy5 = train5.history['accuracy'][-1]
val_accuracy5 = train5.history['val_accuracy'][-1]
print(f'model_5 training accuracy = {training_accuracy5}, \tevaluation accuracy = {val_accuracy5}')
# doubling convolutions/filters of hidden layer significantly improved overall training and validation accuracy surpassing previous values by the 4th epoch vs previous model


# 1st
# add extra convolution layer identical to hidden layer vs previous model
data6 = tf.keras.datasets.cifar10
(training_data, training_labels), (val_data, val_labels) = data6.load_data()
training_images6 = (training_data.reshape(len(training_data), 32, 32, 3)) / 255.0
val_images6 = (val_data.reshape(len(val_data), 32, 32, 3)) / 255.0

m6_callback = myCallback(train5)

m6_layer1 = keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu, input_shape=(32,32,3))
m6_layer2 = keras.layers.MaxPooling2D(2,2)
m6_layer3 = keras.layers.Conv2D(128, (3,3), activation=tf.nn.relu)
m6_layer4 = keras.layers.MaxPooling2D(2,2)
m6_layer5 = keras.layers.Conv2D(128, (3,3), activation=tf.nn.relu)
m6_layer6 = keras.layers.Dense(units=1024, activation=tf.nn.relu)
m6_layer7 = keras.layers.Dense(units=10, activation=tf.nn.softmax)
model_6 = keras.Sequential([m6_layer1, m6_layer2, m6_layer3, m6_layer4, m6_layer5, keras.layers.Flatten(), m6_layer6, m6_layer7])
model_6.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
train6 = model_6.fit(training_images6, training_labels, validation_data=(val_images6, val_labels), callbacks=[m6_callback], epochs=num_epochs)

training_accuracy6 = train6.history['accuracy'][-1]
val_accuracy6 = train6.history['val_accuracy'][-1]
print(f'model_6 training accuracy = {training_accuracy6}, \tevaluation accuracy = {val_accuracy6}')
# additional convolution layer marginally improved validation accuracy moderately decreased training accuracy exhibiting less over fitting tendencies vs previous model


# double convolutions/filters of extra convolutional layers vs previous model
data7 = tf.keras.datasets.cifar10
(training_data, training_labels), (val_data, val_labels) = data7.load_data()
training_images7 = (training_data.reshape(len(training_data), 32, 32, 3)) / 255.0
val_images7 = (val_data.reshape(len(val_data), 32, 32, 3)) / 255.0

m7_callback = myCallback(train6)

m7_layer1 = keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu, input_shape=(32,32,3))
m7_layer2 = keras.layers.MaxPooling2D(2,2)
m7_layer3 = keras.layers.Conv2D(128, (3,3), activation=tf.nn.relu)
m7_layer4 = keras.layers.MaxPooling2D(2,2)
m7_layer5 = keras.layers.Conv2D(256, (3,3), activation=tf.nn.relu)
m7_layer6 = keras.layers.Dense(units=1024, activation=tf.nn.relu)
m7_layer7 = keras.layers.Dense(units=10, activation=tf.nn.softmax)
model_7 = keras.Sequential([m7_layer1, m7_layer2, m7_layer3, m7_layer4, m7_layer5, keras.layers.Flatten(), m7_layer6, m7_layer7])
model_7.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'], callbacks=[m7_callback])
train7 = model_7.fit(training_images7, training_labels, validation_data=(val_images7, val_labels), epochs=num_epochs)

training_accuracy7 = train7.history['accuracy'][-1]
val_accuracy7 = train7.history['val_accuracy'][-1]
print(f'model_7 training accuracy = {training_accuracy7}, \tevaluation accuracy = {val_accuracy7}')
# doubling convolutions/filters of the extra convolution layers marginally reduced training and validation accuracy vs previous model


# reduce convolutions/filters of extra convolution layer by a factor of 1/4 vs previous model
data8 = tf.keras.datasets.cifar10
(training_data, training_labels), (val_data, val_labels) = data8.load_data()
training_images8 = (training_data.reshape(len(training_data), 32, 32, 3)) / 255.0
val_images8 = (val_data.reshape(len(val_data), 32, 32, 3)) / 255.0

m8_callback = myCallback(train7)

m8_layer1 = keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu, input_shape=(32,32,3))
m8_layer2 = keras.layers.MaxPooling2D(2,2)
m8_layer3 = keras.layers.Conv2D(128, (3,3), activation=tf.nn.relu)
m8_layer4 = keras.layers.MaxPooling2D(2,2)
m8_layer5 = keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu)
m8_layer6 = keras.layers.Dense(units=1024, activation=tf.nn.relu)
m8_layer7 = keras.layers.Dense(units=10, activation=tf.nn.softmax)
model_8 = keras.Sequential([m8_layer1, m8_layer2, m8_layer3, m8_layer4, m8_layer5, keras.layers.Flatten(), m8_layer6, m8_layer7])
model_8.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
train8 = model_8.fit(training_images8, training_labels, validation_data=(val_images8, val_labels), callbacks=[m8_callback], epochs=num_epochs)

training_accuracy8 = train8.history['accuracy'][-1]
val_accuracy8 = train8.history['val_accuracy'][-1]
print(f'model_8 training accuracy = {training_accuracy8}, \tevaluation accuracy = {val_accuracy8}')
# reducing convolutions/filters of extra convolution layers noticably reduces training and validation accuracy vs previous model


