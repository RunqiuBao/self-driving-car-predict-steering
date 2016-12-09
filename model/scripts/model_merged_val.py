from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Merge
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, SpatialDropout2D
from keras.utils.visualize_util import plot

left_model = Sequential()
center_model = Sequential()
right_model = Sequential()

#First Convolutional Layer with stride of 2x2 and kernel size 5x5
left_model.add(Convolution2D(24, 5, 5, activation='relu', subsample = (2, 2), border_mode = 'same', input_shape=(3,120,160), init="glorot_uniform", bias = True))
#Second Convolutional Layer with stride of 2x2 and kernel size 5x5
left_model.add(Convolution2D(36, 5, 5, activation='relu', subsample = (2, 2), border_mode = 'same', init="glorot_uniform", bias = True))
#Third Convolutional Layer with stride of 2x2 and kernel size 5x5
left_model.add(Convolution2D(48, 5, 5, activation='relu', subsample = (2, 2), border_mode = 'same', init="glorot_uniform", bias = True))
#Fourth Convolutional Layer with no stride and kernel size 3x3
left_model.add(Convolution2D(64, 3, 3, activation='relu', border_mode = 'same', init="glorot_uniform", bias = True))
#Fifth Convolutional Layer with no stride and kernel size 3x3
left_model.add(Convolution2D(64, 3, 3, activation='relu', border_mode = 'same', init="glorot_uniform", bias = True))
left_model.add(Flatten())

#First Convolutional Layer with stride of 2x2 and kernel size 5x5
center_model.add(Convolution2D(24, 5, 5, activation='relu', subsample = (2, 2), border_mode = 'same', input_shape=(3,120,160), init="glorot_uniform", bias = True))
#Second Convolutional Layer with stride of 2x2 and kernel size 5x5
center_model.add(Convolution2D(36, 5, 5, activation='relu', subsample = (2, 2), border_mode = 'same', init="glorot_uniform", bias = True))
#Third Convolutional Layer with stride of 2x2 and kernel size 5x5
center_model.add(Convolution2D(48, 5, 5, activation='relu', subsample = (2, 2), border_mode = 'same', init="glorot_uniform", bias = True))
#Fourth Convolutional Layer with no stride and kernel size 3x3
center_model.add(Convolution2D(64, 3, 3, activation='relu', border_mode = 'same', init="glorot_uniform", bias = True))
#Fifth Convolutional Layer with no stride and kernel size 3x3
center_model.add(Convolution2D(64, 3, 3, activation='relu', border_mode = 'same', init="glorot_uniform", bias = True))
center_model.add(Flatten())

#First Convolutional Layer with stride of 2x2 and kernel size 5x5
right_model.add(Convolution2D(24, 5, 5, activation='relu', subsample = (2, 2), border_mode = 'same', input_shape=(3,120,160), init="glorot_uniform", bias = True))
#Second Convolutional Layer with stride of 2x2 and kernel size 5x5
right_model.add(Convolution2D(36, 5, 5, activation='relu', subsample = (2, 2), border_mode = 'same', init="glorot_uniform", bias = True))
#Third Convolutional Layer with stride of 2x2 and kernel size 5x5
right_model.add(Convolution2D(48, 5, 5, activation='relu', subsample = (2, 2), border_mode = 'same', init="glorot_uniform", bias = True))
#Fourth Convolutional Layer with no stride and kernel size 3x3
right_model.add(Convolution2D(64, 3, 3, activation='relu', border_mode = 'same', init="glorot_uniform", bias = True))
#Fifth Convolutional Layer with no stride and kernel size 3x3
right_model.add(Convolution2D(64, 3, 3, activation='relu', border_mode = 'same', init="glorot_uniform", bias = True))
right_model.add(Flatten())

merged = Merge([left_model, center_model, right_model], mode='concat')

model = Sequential()
model.add(merged)
model.add(Dense(1164, activation='relu', init="glorot_uniform", bias = True))
model.add(Dense(100, activation='relu', init="glorot_uniform", bias = True))
model.add(Dense(50, activation='relu', init="glorot_uniform", bias = True))
model.add(Dense(10, activation='relu', init="glorot_uniform", bias = True))
model.add(Dense(1))
model.summary()

#plot(left_model, to_file='left_model_val.png')
#plot(center_model, to_file='center_model_val.png')
#plot(right_model, to_file='right_model_val.png')
#plot(model, to_file='model_merged_val.png')

model_json = model.to_json()
with open("model_merged_val.json", "w") as json_file:
    json_file.write(model_json)
