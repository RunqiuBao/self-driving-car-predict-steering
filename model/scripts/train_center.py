#!/usr/bin/python
## Author: akshat
## Date: Dec, 3,2016
# trains the model

from keras.models import model_from_json
import load_data
import numpy
import matplotlib.pyplot as plt
import time

# load json model
json_file = open('model.json', 'r')
loaded_model = json_file.read()
json_file.close()
model = model_from_json(loaded_model)
print "Loaded the model"

# complie the model
model.compile(loss='mse', optimizer='adam')

# set batch_size and number of epochs
batch_size = 500
epochs = 10

# get the values of correct steering angels
y_train_data = load_data.loadY("center")

# training the model
for i in range(epochs):
    trainPredict = numpy.empty([0])
    for j in range(load_data.clen_train/batch_size):
        x_train, y_train = load_data.loadTrainDataC(batch_size)
        history = model.fit(x_train, y_train, nb_epoch = 1, verbose = 2)
        trainPredict = numpy.append(trainPredict, model.predict(x_train))
        print "Epoch" + str(i+1)

    x_train, y_train = load_data.loadTrainDataC(load_data.clen_train - (load_data.ctrain_batch_index%load_data.clen_train))
    history = model.fit(x_train, y_train, nb_epoch = 1, verbose = 2)
    trainPredict = numpy.append(trainPredict, model.predict(x_train))

trainPredict = numpy.expand_dims(trainPredict, axis = 1)

# estimate model performance
trainScore = model.evaluate(x_train, y_train, verbose=2)
print 'Train Score: ', trainScore

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(y_train_data)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[0:len(trainPredict), :] = trainPredict

input_1 = raw_input("Do you want to save the plot? (y/n): ")
input_2 = raw_input("Do you want to save the weights? (y/n): ")

# get time stamp
timestr = time.strftime("%Y%m%d-%H%M%S")

if input_1 == 'y':
    # plot baseline and predictions
    plt.plot(y_train_data, label = 'Dataset')
    plt.plot(trainPredictPlot, label = 'Training Prediction')
    plt.legend(loc = 'upper left')
    plt.savefig('Training_Performance_Center_' + timestr)
    print "Saved plot to disk"

if input_2 == 'y':
    # serialize weights to HDF5
    model.save_weights("weights-center-" + timestr + ".h5")
    print("Saved weights to disk")
