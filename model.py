import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

from data import get_csv_data,generate_samples

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

train,valid = get_csv_data()


model = Sequential()
model.add(Lambda(lambda x:x/255-0.5,input_shape=(32,128,3)))
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(32, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(.25))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))

model.summary()

model.compile(optimizer=Adam(lr=1e-04), loss='mean_squared_error')


batch_size=128

history = model.fit_generator(
    generate_samples(train,batch_size),
    steps_per_epoch=train.shape[0] // batch_size,
    nb_epoch=25,
    validation_data=generate_samples(valid, batch_size),
    validation_steps=valid.shape[0] // batch_size
)

model.save('model.h5')

