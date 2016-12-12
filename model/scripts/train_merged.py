#!/usr/bin/python
## Author: sumanth
## Date: Dec, 8,2016
# trains the sequntial merged model

from keras.models import model_from_json
import load_data
import numpy
import matplotlib.pyplot as plt
import time
import rospkg
import os

#set rospack
#rospack = rospkg.RosPack()
#get package
#data_dir=rospack.get_path('model')
#weights_dir = os.path.join(data_dir, "scripts/outputs/40gb/1.weights-merged-20161209-071920.h5")

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

epochs = 2

# load json model
json_file = open('model_merged.json', 'r')
#json_file = open('model_aicar.json', 'r')
loaded_model = json_file.read()
json_file.close()
model = model_from_json(loaded_model)
print "Loaded the training model"

# loda the pre trained weights
#model.load_weights(weights_dir)
#print "Loaded pre-trained weights.. resuming the training now.."
#model.load_weights("200gb_deg_weights-merged-20161207-232526" + ".h5")
#print "Loaded the pre trained weights"

# complie the model
model.compile(loss='mse', optimizer='adam')

# generator
genT = load_data.trainMDataGen()
genV = load_data.valMDataGen()

# get the values of correct steering angels
y_train_data = load_data.loadY("merged", "validate")

# training the model
print "compiled the model and started training..."
history = model.fit_generator(genT, samples_per_epoch = load_data.clen_train, nb_epoch = epochs, verbose = 1, validation_data = genV, nb_val_samples = load_data.clen_val, max_q_size = 10, nb_worker = 4, pickle_safe = False)

# get time stamp
timestr = time.strftime("%Y%m%d-%H%M%S")

# serialize weights to HDF5
model.save_weights("weights-merged-" + timestr + ".h5")
print("Saved weights to disk")

print "Entering Prediction please wait... Your plots will be generated soon..."

# load json model
json_file = open('model_merged_val.json', 'r')
#json_file = open('model_valaicar.json', 'r')
loaded_model_val = json_file.read()
json_file.close()
model_val = model_from_json(loaded_model_val)
print "Loaded the validation/testing model"

#Load the trained weights
model_val.load_weights("weights-merged-" + timestr + ".h5")

# compile the model
model_val.compile(loss='mse', optimizer='adam')

genT = load_data.testMDataGen()

print 'Started testing..'
trainPredict = model_val.predict_generator(genT, val_samples = load_data.clen_val, max_q_size = 10, nb_worker = 4, pickle_safe = False)

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
plt.savefig('Test_Steering_Angle_merged.jpg')
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
plt.savefig('Loss_Plot_merged.jpg')
print "Saved loss plot to disk"
plt.close()
