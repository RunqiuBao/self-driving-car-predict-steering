#!/usr/bin/python
## Author: sumanth
## Date: Dec, 07,2016
# tests the model

from keras.models import model_from_json
import load_data
import numpy
import matplotlib.pyplot as plt
import time


json_file = open('model_val.json', 'r')
loaded_model_val = json_file.read()
json_file.close()
model_val = model_from_json(loaded_model_val)
print "Loaded the validation/testing model"
#Load the trained weights
model_val.load_weights("200gb_set_weights-center-20161207-152038" + ".h5")

# compile the model
model_val.compile(loss='mse', optimizer='adam')

# get the values of correct steering angels
y_train_data = load_data.loadY("center", "train")
genT = load_data.trainDataGen('center')

trainPredict = model_val.predict_generator(genT, val_samples = load_data.clen_train)

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(y_train_data)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[0:len(trainPredict), :] = trainPredict

#Plotting steering angle actual vs predicted
plt.figure(0)
plt.plot(y_train_data, label = 'Actual Dataset')
plt.plot(trainPredictPlot, label = 'Training Prediction')
plt.title('Steering Angle: Actual vs Predicted')
plt.xlabel('Number of images')
plt.ylabel('Steering angle in radians')
plt.legend(loc = 'upper left')
plt.savefig('Training_Steering_Angle_Center')
print "Saved steering angle plot to disk"
plt.close()
