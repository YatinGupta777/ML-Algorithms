#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 13:42:43 2018

@author: yatingupta
"""

#Part 1 Building the CNN
'''to initialize neural network'''
from keras.models import Sequential

from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#intialising the CNN
classifier = Sequential()

#Step1 - Convolution
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))

#Step2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

'''adding another convolution layer to increase accuracy (making deeper network)'''
classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Step3 flatenning
classifier.add(Flatten())

#Step4 Full connection(making the ANN)
classifier.add(Dense(output_dim = 128,activation = 'relu'))
'''sigmoid function since we have a binary outcome cats or dogs'''
#Output layer
classifier.add(Dense(output_dim = 1,activation = 'sigmoid'))

#Compiling the CNN
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

#Fitting the CNN to images
# Source https://keras.io/preprocessing/image/
from keras.preprocessing.image import ImageDataGenerator
''' Creating more images by rotating them and performing other operations'''
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
            'dataset/training_set',
            target_size=(64, 64),
            batch_size=32,
            class_mode='binary')    

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch = 8000,
        epochs=25,
        validation_data=test_set,
        nb_val_samples=2000)



