import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection
import cv2


def preprocess_image(image):

    image = image[70:140,:,:]
    image = cv2.resize(image,(128,32))

    return image


def get_csv_data():

    CENTER,LEFT,RIGHT,STEERING,THROTTLE,BRAKE,SPEED=range(7)

    driving_log  = pd.io.parsers.read_csv('./SimulatorTraining/driving_log.csv').as_matrix()
    count = len(driving_log)
    ileft = np.random.choice(count,count//2,replace=False)
    iright = np.random.choice(count,count//2,replace=False)

    print(driving_log.shape)

    data=driving_log[:,[CENTER,STEERING,THROTTLE,BRAKE,SPEED]]
    left_data=driving_log[:,[LEFT,STEERING,THROTTLE,BRAKE,SPEED]] [ileft , :]
    right_data=driving_log[:,[RIGHT,STEERING,THROTTLE,BRAKE,SPEED]] [iright , :]

    data=np.concatenate( (data,left_data) )
    data=np.concatenate( (data,right_data) )
    train, valid  = model_selection.train_test_split(data, test_size=.2)

    return train,valid


def get_batch_data(batch):
    IMG,STEERING,THROTTLE,BRAKE,SPEED=range(5)
    x,y = [],[]
    for row in batch:
        image = preprocess_image(plt.imread( "SimulatorTraining/" + row[IMG]))
        angle = row[STEERING]
        x.append( image )
        y.append( angle )
        x.append( image[:,::-1,:] )
        y.append( -1 * angle )

    return np.array(x),np.array(y)

def generate_samples(data,batch_size=128):
    while True:
        for bnext in range(0, len(data), batch_size):
            batch=data[bnext:bnext+batch_size]
            x,y=get_batch_data(batch)
            yield (x,y)

