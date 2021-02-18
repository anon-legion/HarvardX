# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 11:52:46 2021

@author: =GV=
"""

try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass

!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip \
    -O /tmp/horse-or-human.zip

!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip \
    -O /tmp/validation-horse-or-human.zip

import os
import zipfile

local_zip = '/tmp/horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/horse-or-human')
local_zip = '/tmp/validation-horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/validation-horse-or-human')
zip_ref.close()

# Directory with our training horse pictures
train_horse_dir = os.path.join('/tmp/horse-or-human/horses')
# Directory with our training human pictures
train_human_dir = os.path.join('/tmp/horse-or-human/humans')
# Directory with our training horse pictures
validation_horse_dir = os.path.join('/tmp/validation-horse-or-human/horses')
# Directory with our training human pictures
validation_human_dir = os.path.join('/tmp/validation-horse-or-human/humans')
train_horse_names = os.listdir('/tmp/horse-or-human/horses')
print(train_horse_names[:10])
train_human_names = os.listdir('/tmp/horse-or-human/humans')
print(train_human_names[:10])
validation_horse_hames = os.listdir('/tmp/validation-horse-or-human/horses')
print(validation_horse_hames[:10])
validation_human_names = os.listdir('/tmp/validation-horse-or-human/humans')
print(validation_human_names[:10])


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
train_gen = train_datagen.flow_from_directory('/tmp/horse-or-human/', target_size=(300,300), batch_size=((len(train_horse_names) + len(train_human_names))//num_steps), class_mode='binary')
val_datagen = ImageDataGenerator(rescale=1./255)
val_gen = val_datagen.flow_from_directory('/tmp/validation-horse-or-human', target_size=(300,300), batch_size=((len(validation_horse_hames) + len(validation_human_names))//num_steps), class_mode='binary')

num_epochs = 9
num_steps = 8

m1_layer1 = keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu, input_shape=(300,300,3))
m1_layer2 = keras.layers.MaxPooling2D(2,2)
m1_layer3 = keras.layers.Conv2D(128, (3,3), activation=tf.nn.relu)
m1_layer4 = keras.layers.MaxPooling2D(2,2)
m1_layer5 = keras.layers.Conv2D(128, (3,3), activation=tf.nn.relu)
m1_layer6 = keras.layers.MaxPooling2D(2,2)
m1_layer7 = keras.layers.Dense(units=256, activation=tf.nn.relu)
m1_layer8 = keras.layers.Dense(units=1, activation='sigmoid')
model_1 = keras.Sequential([m1_layer1, m1_layer2, m1_layer3, m1_layer4, m1_layer5, m1_layer6, keras.layers.Flatten(), m1_layer7, m1_layer8])
model_1.compile(optimizer=keras.optimizers.RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])
train_m1 = model_1.fit(train_gen, validation_data=val_gen, validation_steps = 8, steps_per_epoch=8, epochs=num_epochs)

train_accuracy1 = train_m1.history['acc'][-1]
val_accuracy1 = train_m1.history['val_acc'][-1]
print(f'model_1 training accuracy = {train_accuracy1},\tvalidation accuracy = {val_accuracy1}')


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# add extra identical dense layer before output layer
train_datagen2 = ImageDataGenerator(rescale=1./255)
train_gen2 = train_datagen2.flow_from_directory('/tmp/horse-or-human/', target_size=(300,300), batch_size=((len(train_horse_names) + len(train_human_names))//num_steps), class_mode='binary')
val_datagen2 = ImageDataGenerator(rescale=1./255)
val_gen2 = val_datagen2.flow_from_directory('/tmp/validation-horse-or-human', target_size=(300,300), batch_size=((len(validation_horse_hames) + len(validation_human_names))//num_steps), class_mode='binary')

class myCallback(tf.keras.callbacks.Callback):
  def __init__(self, other):
    self.other = other
  def on_epoch_end(self, epoch, logs={}):
    if logs.get('acc') > self.other.history['acc'][-1] and logs.get('val_acc') > self.other.history['val_acc'][-1]:
      self.model.stop_training = True

callback_m2 = myCallback(train_m1)

m2_layer1 = keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu, input_shape=(300,300,3))
m2_layer2 = keras.layers.MaxPooling2D(2,2)
m2_layer3 = keras.layers.Conv2D(128, (3,3), activation=tf.nn.relu)
m2_layer4 = keras.layers.MaxPooling2D(2,2)
m2_layer5 = keras.layers.Conv2D(128, (3,3), activation=tf.nn.relu)
m2_layer6 = keras.layers.MaxPooling2D(2,2)
m2_layer7 = keras.layers.Dense(units=256, activation=tf.nn.relu)
m2_layer8 = keras.layers.Dense(units=256, activation=tf.nn.relu)
m2_layer9 = keras.layers.Dense(units=1, activation='sigmoid')
model_2 = keras.Sequential([m2_layer1, m2_layer2, m2_layer3, m2_layer4, m2_layer5, m2_layer6, keras.layers.Flatten(), m2_layer7, m2_layer8, m2_layer9])
model_2.compile(optimizer=keras.optimizers.RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])
train_m2 = model_2.fit(train_gen2, validation_data=val_gen2, validation_steps=8, steps_per_epoch=8, epochs=num_epochs, callbacks=[callback_m2])

train_accuracy2 = train_m2.history['acc'][-1]
val_accuracy2 = train_m2.history['val_acc'][-1]
print(f'model_2 training accuracy = {train_accuracy2},\tvalidation accuracy = {val_accuracy2}')
# extra identical dense layer significantly reduced overall training and validation accuracy (~9% and ~11% respectively) vs previous model


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# reduce neurons of extra dense layer by a factor of 1/2
train_datagen3 = ImageDataGenerator(rescale=1./255)
train_gen3 = train_datagen3.flow_from_directory('/tmp/horse-or-human/', target_size=(300,300), batch_size=((len(train_horse_names) + len(train_human_names))//num_steps), class_mode='binary')
val_datagen3 = ImageDataGenerator(rescale=1./255)
val_gen3 = val_datagen3.flow_from_directory('/tmp/validation-horse-or-human', target_size=(300,300), batch_size=((len(validation_horse_hames) + len(validation_human_names))//num_steps), class_mode='binary')

callback_m3 = myCallback(train_m2)

m3_layer1 = keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu, input_shape=(300,300,3))
m3_layer2 = keras.layers.MaxPooling2D(2,2)
m3_layer3 = keras.layers.Conv2D(128, (3,3), activation=tf.nn.relu)
m3_layer4 = keras.layers.MaxPooling2D(2,2)
m3_layer5 = keras.layers.Conv2D(128, (3,3), activation=tf.nn.relu)
m3_layer6 = keras.layers.MaxPooling2D(2,2)
m3_layer7 = keras.layers.Dense(units=256, activation=tf.nn.relu)
m3_layer8 = keras.layers.Dense(units=128, activation=tf.nn.relu)
m3_layer9 = keras.layers.Dense(units=1, activation='sigmoid')
model_3 = keras.Sequential([m3_layer1,m3_layer2, m3_layer3, m3_layer4, m3_layer5, m3_layer6, keras.layers.Flatten(), m3_layer7, m3_layer8, m3_layer9])
model_3.compile(optimizer=keras.optimizers.RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])
train_m3 = model_3.fit(train_gen3, validation_data=val_gen3, validation_steps=8, steps_per_epoch=8, epochs=num_epochs, callbacks=[callback_m3])

train_accuracy3 = train_m3.history['acc'][-1]
val_accuracy3 = train_m3.history['val_acc'][-1]
print(f'model_3 training accuracy = {train_accuracy3},\tvalidation accuracy = {val_accuracy3}')
# reduced neurons in the extra dense layer significantly increased overall training and validation accuracy surpassing values by the 6th epoch vs previous model


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# increase neurons of extra dense layer by a factor of 4 vs previous model or by 2 vs model_2
train_datagen4 = ImageDataGenerator(rescale=1./255)
train_gen4 = train_datagen4.flow_from_directory('/tmp/horse-or-human/', target_size=(300,300), batch_size=((len(train_horse_names) + len(train_human_names))//num_steps), class_mode='binary')
val_datagen4 = ImageDataGenerator(rescale=1./255)
val_gen4 = val_datagen4.flow_from_directory('/tmp/validation-horse-or-human', target_size=(300,300), batch_size=((len(validation_horse_hames) + len(validation_human_names))//num_steps), class_mode='binary')

callback_m4 = myCallback(train_m3)

m4_layer1 = keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu, input_shape=(300,300,3))
m4_layer2 = keras.layers.MaxPooling2D(2,2)
m4_layer3 = keras.layers.Conv2D(128, (3,3), activation=tf.nn.relu)
m4_layer4 = keras.layers.MaxPooling2D(2,2)
m4_layer5 = keras.layers.Conv2D(128, (3,3), activation=tf.nn.relu)
m4_layer6 = keras.layers.MaxPooling2D(2,2)
m4_layer7 = keras.layers.Dense(units=256, activation=tf.nn.relu)
m4_layer8 = keras.layers.Dense(units=512, activation=tf.nn.relu)
m4_layer9 = keras.layers.Dense(units=1, activation='sigmoid')
model_4 = keras.Sequential([m4_layer1, m4_layer2, m4_layer3, m4_layer4, m4_layer5, m4_layer6, keras.layers.Flatten(), m4_layer7, m4_layer8, m4_layer9])
model_4.compile(optimizer=keras.optimizers.RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])
train_m4 = model_4.fit(train_gen4, validation_data=val_gen4, validation_steps=8, steps_per_epoch=8, epochs=num_epochs, callbacks=[callback_m4])

train_accuracy4 = train_m4.history['acc'][-1]
val_accuracy4 = train_m4.history['val_acc'][-1]
print(f'model_4 training accuracy = {train_accuracy4},\tvalidation accuracy = {val_accuracy4}')
# increased neurons of extra dense layer significantly reduced overall training and validation accuracy vs previous model
# increased neurons of extra dense layer reduced overall training and validation accuracy (~2% and ~4% delta respectively) vs model_2


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# reduce neurons of extra dense layer to identical number with other dense layer
# add extra identical convolution and pooling layer to the last convolution layer
train_datagen5 = ImageDataGenerator(rescale=1./255)
train_gen5 = train_datagen5.flow_from_directory('/tmp/horse-or-human/', target_size=(300,300), batch_size=((len(train_horse_names) + len(train_human_names))//num_steps), class_mode='binary')
val_datagen5 = ImageDataGenerator(rescale=1./255)
val_gen5 = val_datagen5.flow_from_directory('/tmp/validation-horse-or-human', target_size=(300,300), batch_size=((len(validation_horse_hames) + len(validation_human_names))//num_steps), class_mode='binary')

callback_m5 = myCallback(train_m4)

m5_layer1 = keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu, input_shape=(300,300,3))
m5_layer2 = keras.layers.MaxPooling2D(2,2)
m5_layer3 = keras.layers.Conv2D(128, (3,3), activation=tf.nn.relu)
m5_layer4 = keras.layers.MaxPooling2D(2,2)
m5_layer5 = keras.layers.Conv2D(128, (3,3), activation=tf.nn.relu)
m5_layer6 = keras.layers.MaxPooling2D(2,2)
m5_layer7 = keras.layers.Conv2D(128, (3,3), activation=tf.nn.relu)
m5_layer8 = keras.layers.MaxPooling2D(2,2)
m5_layer9 = keras.layers.Dense(units=256, activation=tf.nn.relu)
m5_layer10 = keras.layers.Dense(units=256, activation=tf.nn.relu)
m5_layer11 = keras.layers.Dense(units=1, activation='sigmoid')
model_5 = keras.Sequential([m5_layer1,m5_layer2, m5_layer3, m5_layer4, m5_layer5, m5_layer6, m5_layer7, m5_layer8, keras.layers.Flatten(), m5_layer9, m5_layer10, m5_layer11])
model_5.compile(optimizer=keras.optimizers.RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])
train_m5 = model_5.fit(train_gen5, validation_data=val_gen5, validation_steps=8, steps_per_epoch=8, epochs=num_epochs, callbacks=[callback_m5])

train_accuracy5 = train_m5.history['acc'][-1]
val_accuracy5 = train_m5.history['val_acc'][-1]
print(f'model_5 training accuracy = {train_accuracy5},\tvalidation accuracy = {val_accuracy5}')
# extra identical convolution and pooling layers increased overall training and validation accuracy (~3% delta respectively) vs previous model
# extra identical convolution and pooling layers marginally increased overall training and validation accuracy (~1% and ~6% delta respectively) vs model_2


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# double the convolutions of extra convolution layer vs previous model
callback_m6 = myCallback(train_m5)

train_datagen6 = ImageDataGenerator(rescale=1./255)
train_gen6 = train_datagen.flow_from_directory('/tmp/horse-or-human/', target_size=(300,300), batch_size=((len(train_horse_names) + len(train_human_names))//num_steps), class_mode='binary')
val_datagen6 = ImageDataGenerator(rescale=1./255)
val_gen6 = val_datagen6.flow_from_directory('/tmp/validation-horse-or-human/', target_size=(300,300), batch_size=((len(validation_horse_hames) + len(validation_human_names))//num_steps), class_mode='binary')

m6_layer1 = keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu, input_shape=(300,300,3))
m6_layer2 = keras.layers.MaxPooling2D(2,2)
m6_layer3 = keras.layers.Conv2D(128, (3,3), activation=tf.nn.relu)
m6_layer4 = keras.layers.MaxPooling2D(2,2)
m6_layer5 = keras.layers.Conv2D(128, (3,3), activation=tf.nn.relu)
m6_layer6 = keras.layers.MaxPooling2D(2,2)
m6_layer7 = keras.layers.Conv2D(256, (3,3), activation=tf.nn.relu)
m6_layer8 = keras.layers.MaxPooling2D(2,2)
m6_layer9 = keras.layers.Dense(units=256, activation=tf.nn.relu)
m6_layer10 = keras.layers.Dense(units=256, activation=tf.nn.relu)
m6_layer11 = keras.layers.Dense(units=1, activation='sigmoid')
model_6 = keras.Sequential([m6_layer1, m6_layer2, m6_layer3, m6_layer4, m6_layer5, m6_layer6, m6_layer7, m6_layer8, keras.layers.Flatten(), m6_layer9, m6_layer10, m6_layer11])
model_6.compile(optimizer=keras.optimizers.RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])
train_m6 = model_6.fit(train_gen6, validation_data=val_gen6, validation_steps=num_steps, steps_per_epoch=num_steps, epochs=num_epochs, callbacks=[callback_m6])

train_accuracy6 = train_m6.history['acc'][-1]
val_accuracy6 = train_m6.history['val_acc'][-1]
print(f'model_6 training accuracy = {train_accuracy6},\tvalidation accuracy = {val_accuracy6}')
# doubling convolutions/filters significantly reduced overall training and validation accuracy (~9% and ~13% delat respectively) vs previous model


# test model
import numpy as np
from google.colab import files
from keras.preprocessing import image

uploaded = files.upload()

for fn in uploaded.keys():
 
  # predicting images
  path = '/content/' + fn
  img = image.load_img(path, target_size=(300, 300))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  image_tensor = np.vstack([x])
  classes = model.predict(image_tensor) # replace model with modelName
  print(classes)
  print(classes[0])
  if classes[0]>0.5:
    print(fn + " is a human")
  else:
    print(fn + " is a horse")
    

import os, signal
os.kill(os.getpid(), signal.SIGKILL)

