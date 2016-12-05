from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, SpatialDropout2D
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D

model = Sequential()

#First Convolutional Layer with stride of 2x2 and kernel size 5x5
model.add(Convolution2D(24, 5, 5, activation='relu', subsample = (2, 2), border_mode = 'same', input_shape=(3,120,160)))
model.add(SpatialDropout2D(0.5))

#Second Convolutional Layer with stride of 2x2 and kernel size 5x5
model.add(Convolution2D(36, 5, 5, activation='relu', subsample = (2, 2), border_mode = 'same'))
model.add(SpatialDropout2D(0.5))

#Third Convolutional Layer with stride of 2x2 and kernel size 5x5
model.add(Convolution2D(48, 5, 5, activation='relu', subsample = (2, 2), border_mode = 'same'))
model.add(SpatialDropout2D(0.5))

#Fourth Convolutional Layer with no stride and kernel size 3x3
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode = 'same'))
model.add(SpatialDropout2D(0.5))

#Fifth Convolutional Layer with no stride and kernel size 3x3
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode = 'same'))

model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='tanh'))
model.summary()

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
