from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, SpatialDropout2D
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D

model = Sequential()

#First Convolutional Layer with stride of 2x2 and kernel size 5x5
model.add(Convolution2D(24, 5, 5, activation='relu', subsample = (2, 2), border_mode = 'same', input_shape=(3,120,160), init="glorot_uniform", bias = True, name = 'conv1'))

#Second Convolutional Layer with stride of 2x2 and kernel size 5x5
model.add(Convolution2D(36, 5, 5, activation='relu', subsample = (2, 2), border_mode = 'same', init="glorot_uniform", bias = True, name = 'conv2'))

#Third Convolutional Layer with stride of 2x2 and kernel size 5x5
model.add(Convolution2D(48, 5, 5, activation='relu', subsample = (2, 2), border_mode = 'same', init="glorot_uniform", bias = True, name = 'conv3'))

#Fourth Convolutional Layer with no stride and kernel size 3x3
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode = 'same', init="glorot_uniform", bias = True, name = 'conv4'))

#Fifth Convolutional Layer with no stride and kernel size 3x3
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode = 'same', init="glorot_uniform", bias = True, name = 'conv5'))

model.add(Flatten())
model.add(Dense(1164, activation='relu', init="glorot_uniform", bias = True, name = 'dense0'))
model.add(Dense(100, activation='relu', init="glorot_uniform", bias = True, name = 'dense1'))
model.add(Dense(50, activation='relu', init="glorot_uniform", bias = True, name = 'dense2'))
model.add(Dense(10, activation='relu', init="glorot_uniform", bias = True, name = 'dense3'))
model.add(Dense(1, name = 'output'))
model.summary()

model_json = model.to_json()
with open("model_val.json", "w") as json_file:
    json_file.write(model_json)
