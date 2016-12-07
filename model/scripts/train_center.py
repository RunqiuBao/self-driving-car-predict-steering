#!/usr/bin/python
## Author: akshat
## Date: Dec, 3,2016
# trains the model

from keras.models import model_from_json
import load_data
import numpy
import matplotlib.pyplot as plt
import time

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

epochs = 5

# load json model
json_file = open('model.json', 'r')
#json_file = open('model_aicar.json', 'r')
loaded_model = json_file.read()
json_file.close()
model = model_from_json(loaded_model)
print "Loaded the training model"

# loda the pre trained weights
# model.load_weights("200gb_set_weights-center-20161207-152038" + timestr + ".h5")
# print "Loaded the pre trained weights"

# complie the model
model.compile(loss='mse', optimizer='adam')

# generator
genT = load_data.trainDataGen('center')
genV = load_data.valDataGen('center')

# get the values of correct steering angels
y_train_data = load_data.loadY("center", "validate")

# training the model
print "compiled the model and started training..."
history = model.fit_generator(genT, samples_per_epoch = load_data.clen_train, nb_epoch = epochs, verbose = 1, validation_data = genV, nb_val_samples = load_data.clen_val)

# get time stamp
timestr = time.strftime("%Y%m%d-%H%M%S")

# serialize weights to HDF5
model.save_weights("weights-center-" + timestr + ".h5")
print("Saved weights to disk")

print "Entering Prediction please wait... Your plots will be generated soon..."

# load json model
json_file = open('model_val.json', 'r')
#json_file = open('model_valaicar.json', 'r')
loaded_model_val = json_file.read()
json_file.close()
model_val = model_from_json(loaded_model_val)
print "Loaded the validation/testing model"

#Load the trained weights
model_val.load_weights("weights-center-" + timestr + ".h5")

# compile the model
model_val.compile(loss='mse', optimizer='adam')

genT = load_data.testDataGen('center')

trainPredict = model_val.predict_generator(genT, val_samples = load_data.clen_val)

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
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='upper left')
plt.savefig('Loss_Plot_Center')
print "Saved loss plot to disk"
plt.close()
