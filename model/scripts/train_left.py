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
y_train_data = load_data.loadY()

# training the model
for i in range(epochs):
    trainPredict = numpy.empty([0])
    for j in range(load_data.len_train_samples/batch_size):
        x_train, y_train = load_data.loadTrainData(batch_size, 'left')
        history = model.fit(x_train, y_train, nb_epoch = 1, verbose = 2)
        trainPredict = numpy.append(trainPredict, model.predict(x_train))
        print "Epoch" + str(i+1)

    x_train, y_train = load_data.loadTrainData(load_data.len_train_samples - (load_data.train_batch_index%load_data.len_train_samples), 'left')
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

# plot baseline and predictions
plt.plot(y_train_data, label = 'Dataset')
plt.plot(trainPredictPlot, label = 'Training Prediction')
plt.legend(loc = 'upper left')
plt.savefig('Performance')
plt.show()

# get time stamp
timestr = time.strftime("%Y%m%d-%H%M%S")
# serialize weights to HDF5
model.save_weights("weights-left-" + timestr + ".h5")
print("Saved weights to disk")
