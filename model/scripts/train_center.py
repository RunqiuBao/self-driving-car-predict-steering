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
epochs = 200

# get the values of correct steering angels
y_train_data = load_data.loadY("center")

#Empty array to record loss at the end of every epoch
loss = numpy.empty([0])

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
    loss = numpy.append(loss, history.history['loss'])
    print "Epoch" + str(i+1)

    trainPredict = numpy.expand_dims(trainPredict, axis = 1)

    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(y_train_data)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[0:len(trainPredict), :] = trainPredict

    #Plotting steering angle actual vs predicted
    plt.figure(0)
    plt.plot(y_train_data, label = 'Actual Dataset')
    plt.plot(trainPredictPlot, label = 'Training Prediction')
    plt.title('Steering Angle: Actual vs Predicted for epoch ' + str(i + 1))
    plt.xlabel('Number of images')
    plt.ylabel('Steering angle in radians')
    plt.legend(loc = 'upper left')
    plt.savefig('Training_Steering_Angle_Center_epoch_' + str(i + 1))
    print "Saved steering angle plot to disk"
    plt.close()

# get time stamp
timestr = time.strftime("%Y%m%d-%H%M%S")

#Plot loss
plt.figure(1)
plt.plot(loss, label = 'Loss')
plt.title('Change in Loss over number of epochs')
plt.xlabel('Number of epochs')
plt.ylabel('Loss')
plt.legend(loc = 'upper left')
plt.savefig('Training_Loss_Center_' + timestr)
print "Saved loss plot to disk"
plt.close()

# serialize weights to HDF5
model.save_weights("weights-center-" + timestr + ".h5")
print("Saved weights to disk")
