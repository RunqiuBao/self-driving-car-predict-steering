#!/usr/bin/python
## Author: sumanth
## Date: Nov, 28,2016
# loads the data into keras

import os
import numpy
import pandas
import rospkg
import cv2

train_size = 0.8
val_size = 0.2

# global index for the data
train_batch_index = 0
val_batch_index = 0

#set rospack
rospack = rospkg.RosPack()
#get package
data_dir=rospack.get_path('dataset')
csv_dir = os.path.join(data_dir, "yaml_files")

if not os.path.exists(csv_dir):
    print "csv directory doesnt exist"


csv_file=os.path.join(csv_dir, 'final_interpolated.csv')
if not os.path.exists(csv_file):
    print "csv file doesnt exist"

#fetch the data from csv
data_inputs = pandas.read_csv(csv_file, usecols=['filename'], engine='python', skipfooter=0)
data_labels = pandas.read_csv(csv_file, usecols=['angle'], engine='python', skipfooter=0)

if not(len(data_inputs.values) and len(data_labels.values)):
    print "error in dataset"

# split into train and test datas
train_x = data_inputs[:int(len(data_inputs.values)*train_size)]
train_y = data_labels[:int(len(data_labels.values)*train_size)]

val_x = data_inputs[-int(len(data_inputs.values)*val_size):]
val_y = data_labels[-int(len(data_labels.values)*val_size):]

len_train_samples = len(train_x)
len_val_samples = len(val_x)

def loadY():
    return train_y.values[:, 0]

def loadTrainData(batch_size):
    global train_batch_index
    train_x = []
    train_y = []
    # fetch all the images and the labels
    for i in range(0,batch_size):
        img_file=os.path.join(data_dir, data_inputs.values[(train_batch_index + i) % len_train_samples][0][3:])
        x = cv2.imread(img_file)
        # normalise the image
        xt = cv2.resize(x.copy()/255.0, (640, 480)).astype(numpy.float32)
        xt = xt.transpose((2, 0, 1))
        #xt = numpy.expand_dims(xt, axis = 0)
        train_x.append(xt)

        # as the steering wheel angle is proportional to inverse of turning radius
        # we directly use the steering wheel angle (source: NVIDIA uses the inverse of turning radius)
        # but converted to radians

        train_y.append(data_labels.values[(train_batch_index + i) % len_train_samples][0])
    train_x = numpy.array(train_x)
    train_y = numpy.expand_dims(train_y, axis = 1)
    #train_x = [train_x]
    #increment the index
    train_batch_index += batch_size

    return train_x, train_y

def loadValData(batch_size):
    global val_batch_index
    val_x = []
    val_y = []
    # fetch all the images and the labels
    for i in range(0,batch_size):
        img_file=os.path.join(data_dir, data_inputs.values[(val_batch_index + i) % len_val_samples][0][3:])
        x = cv2.imread(img_file)
        # normalise the image
        xt = cv2.resize(x.copy()/255.0, (640, 480)).astype(numpy.float32)
        xt = xt.transpose((2, 0, 1))
        xt = numpy.expand_dims(xt, axis = 0)
        val_x.append(xt)

        # as the steering wheel angle is proportional to inverse of turning radius
        # we directly use the steering wheel angle (source: NVIDIA uses the inverse of turning radius)
        # but converted to radians
        val_y.append(data_labels.values[(val_batch_index + i) % len_val_samples][0])

    #increment the index
    val_batch_index += batch_size

    return val_x, val_y
