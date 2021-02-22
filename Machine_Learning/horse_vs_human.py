# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 10:37:30 2021

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

from tensorflow import keras, nn
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from google.colab import files
import os
import zipfile
import signal
import numpy as np


train_zip = '/tmp/horse-or-human.zip'
val_zip = '/tmp/validation-horse-or-human.zip'
train_ref = zipfile.ZipFile(train_zip, 'r')
train_ref.extractall('/tmp/horse-or-human')
train_ref.close()
val_ref = zipfile.ZipFile(val_zip, 'r')
val_ref.extractall('/tmp/validation-horse-or-human')
val_ref.close()

train_dir = os.path.join('/tmp/horse-or-human')
val_dir = os.path.join('/tmp/validation-horse-or-human')
train_imgcount = len(os.listdir(os.path.join(train_dir, 'horses'))) + len(os.listdir(os.path.join(train_dir, 'humans')))
val_imgcount = len(os.listdir(os.path.join(val_dir, 'horses'))) + len(os.listdir(os.path.join(val_dir, 'humans')))
print(f'train count = {train_imgcount}')
print(f'val count = {val_imgcount}')


layer1 = keras.layers.Conv2D(64, (3,3), activation=nn.relu, input_shape=(300,300,3))
layer2 = keras.layers.MaxPooling2D(2,2)
layer3 = keras.layers.Conv2D(128, (3,3), activation=nn.relu)
layer4 = keras.layers.MaxPooling2D(2,2)
layer5 = keras.layers.Conv2D(256, (3,3), activation=nn.relu)
layer6 = keras.layers.MaxPooling2D(2,2)
layer7 = keras.layers.Conv2D(256, (3,3), activation=nn.relu)
layer8 = keras.layers.Dense(units=256, activation=nn.relu)
layer9 = keras.layers.Dropout(0.2)
layer10 = keras.layers.Dense(units=512, activation=nn.relu)
layer11 = keras.layers.Dropout(0.2)
layer12 = keras.layers.Dense(units=512, activation=nn.relu)
layer13 = keras.layers.Dense(units=1, activation='sigmoid')
model = keras.Sequential([layer1, layer2, layer3, layer4, layer5, layer6, layer7, keras.layers.Flatten(), layer8, layer9, layer10, layer11, layer12, layer13])
model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

num_epochs = 20
num_steps = 9
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=33, height_shift_range=0.2, width_shift_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_gen = train_datagen.flow_from_directory(train_dir, target_size=(300,300), batch_size=(train_imgcount // num_steps), class_mode='binary')
val_datagen = ImageDataGenerator(rescale=1./255)
val_gen = val_datagen.flow_from_directory(val_dir, target_size=(300,300), batch_size=(val_imgcount // num_steps), class_mode='binary')

train = model.fit(train_gen, validation_data=val_gen, validation_steps=num_steps, steps_per_epoch=num_steps, epochs=num_epochs)

train_accuracy = train.history['accuracy'][-1]
val_accuracy = train.history['val_accuracy'][-1]
print(f'model training accuracy = {train_accuracy},\tvalidation accuracy = {val_accuracy}')


while True:
  test = input('do you wish to test the model? (y/n):\n')
  if test.lower() == 'n':
    break
  elif test.lower() == 'y':
    file = files.upload()
    for fn in file.keys():
      path = '/content/' + fn
      img = image.load_img(path, target_size=(300,300))
      x = image.img_to_array(img)
      x = np.expand_dims(x, axis=0)

      image_tensor = np.vstack([x])
      classes = model.predict(image_tensor)
      print(classes)
      print(classes[0])
      if classes[0] > 0.5:
        print(f'{fn} is a human')
      else:
        print(f'{fn} is a horse')
  else:
    print('invalid input, try again!')
    

os.kill(os.getpid(), signal.SIGKILL)