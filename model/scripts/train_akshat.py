from keras.models import model_from_json
import load_data
import numpy
import matplotlib.pyplot as plt

# load json model
json_file = open('model.json', 'r')
loaded_model = json_file.read()
json_file.close()
model = model_from_json(loaded_model)
print "Loaded the model"

model.compile(loss='mse', optimizer='adam')

batch_size = 100
epochs = 5

#training_history = []

y_train_data = load_data.loadY()
trainPredict = []

for i in range(epochs):
    for j in range(load_data.len_train_samples/batch_size):
        x_train, y_train = load_data.loadTrainData(batch_size)
        history = model.fit(x_train, y_train, nb_epoch = 1, verbose = 2)
        #print history.history.keys()
        # Estimate model performance
        #trainScore = model.evaluate(x_train, y_train, verbose=0)
        #print 'Train Score: ', trainScore
        # generate predictions for training
        trainPredict.append(model.predict(x_train))
        #print 'Prediction: ', trainPredict
        # shift train predictions for plotting

# Estimate model performance
trainScore = model.evaluate(x_train, y_train, verbose=2)
print 'Train Score: ', trainScore

trainPredictPlot = numpy.empty_like(y_train_data)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[0:len(trainPredict), :] = trainPredict
# plot baseline and predictions
plt.plot(y_train_data, label = 'Dataset')
plt.plot(trainPredictPlot, label = 'Training Prediction')
plt.legend(loc = 'upper left')
plt.show()

# serialize weights to HDF5
model.save_weights("model_epochs_1-5.h5")
print("Saved model to disk")
