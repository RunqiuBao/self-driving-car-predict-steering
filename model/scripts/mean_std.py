import load_data
import os
import sys
import rospkg
from keras.models import model_from_json

file_list = []

#set rospack
rospack = rospkg.RosPack()
#get package
data_dir=rospack.get_path('model')
hf5_file = os.path.join(data_dir, "scripts//200gb_set/in_deg/")

#get all the weights
for file in [doc for doc in os.listdir(hf5_file)
if doc.endswith(".h5")]:
    file_list.append(file)

for i in file_list:
    #Load the trained weights
    json_file = open('model_val.json', 'r')
    loaded_model_val = json_file.read()
    json_file.close()
    model_val = model_from_json(loaded_model_val)
    print "loaded the validation model"
    model_val.load_weights(hf5_file[i])
    print "loaded the weights", i
    model_val.compile(loss='mse', optimizer='adam')
    print "compiled the model"

    # get the values of correct steering angels
    y_train_data = load_data.loadY("center", "train")
    genT = load_data.trainDataGen('center')

    genT = load_data.trainDataGen('center')
    trainPredict = model_val.predict_generator(genT, val_samples = load_data.clen_train)
