#!/usr/bin/python
## Author: akshat
## Date: Dec, 3,2016
# trains the model

from keras.models import model_from_json
import load_data
import numpy
import matplotlib.pyplot as plt
import time

epochs = 50

# load json model
json_file = open('model.json', 'r')
loaded_model = json_file.read()
json_file.close()
model = model_from_json(loaded_model)
print "Loaded the training model"

# complie the model
model.compile(loss='mse', optimizer='adam')

# generator
genT = load_data.trainDataGen('center')
genV = load_data.valDataGen('center')

# get the values of correct steering angels
y_train_data = load_data.loadY("center", "validate")

# training the model
history = model.fit_generator(genT, samples_per_epoch = load_data.clen_train, nb_epoch = epochs, verbose = 1, nb_worker = 1)
print "Entering Prediction please wait... Your plots will be generated soon..."
# load json model
json_file = open('model_val.json', 'r')
loaded_model_val = json_file.read()
json_file.close()
model_val = model_from_json(loaded_model_val)
print "Loaded the validation/testing model"
trainPredict = model_val.predict_generator(genV, val_samples = load_data.clen_val, nb_worker = 1)

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

# summarize history for loss
plt.figure(1)
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.savefig('Loss_Plot_Center')
print "Saved loss plot to disk"
plt.close()

# get time stamp
timestr = time.strftime("%Y%m%d-%H%M%S")

# serialize weights to HDF5
model.save_weights("weights-center-" + timestr + ".h5")
print("Saved weights to disk")
