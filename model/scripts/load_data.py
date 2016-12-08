#!/usr/bin/python
## Author: sumanth
## Date: Nov, 28,2016
# loads the data into keras

import os
import numpy
import pandas
import rospkg
import cv2
import random

train_size = 0.8
val_size = 0.2
batch_size = 200

# global index for the data
mtrain_batch_index = 0
mval_batch_index = 0
ltrain_batch_index = 0
lval_batch_index = 0
rtrain_batch_index = 0
rval_batch_index = 0
ctrain_batch_index = 0
cval_batch_index = 0
ltest_batch_index = 0
rtest_batch_index = 0
ctest_batch_index = 0
mdata_batch_index = 0
mrval_batch_index = 0
mtrain_batch_index = 0

#set rospack
rospack = rospkg.RosPack()
#get package
data_dir=rospack.get_path('dataset')
csv_dir = os.path.join(data_dir, "yaml_files")

if not os.path.exists(csv_dir):
    print "csv directory doesnt exist"

mimages_csv_file=os.path.join(csv_dir, 'final_interpolated.csv')
limages_csv_file=os.path.join(csv_dir, 'left_camera_image.csv')
rimages_csv_file=os.path.join(csv_dir, 'right_camera_image.csv')
cimages_csv_file=os.path.join(csv_dir, 'center_camera_image.csv')

if not (os.path.exists(mimages_csv_file) or os.path.exists(limages_csv_file) or os.path.exists(rimages_csv_file) or os.path.exists(cimages_csv_file)):
    print "csv file doesnt exist"

#fetch the data from csv
m_inputs = pandas.read_csv(mimages_csv_file, usecols=['filename'], engine='python', skipfooter=0)
m_labels = pandas.read_csv(mimages_csv_file, usecols=['angle'], engine='python', skipfooter=0)

l_inputs = pandas.read_csv(limages_csv_file, usecols=['filename'], engine='python', skipfooter=0)
l_labels = pandas.read_csv(limages_csv_file, usecols=['angle'], engine='python', skipfooter=0)

r_inputs = pandas.read_csv(rimages_csv_file, usecols=['filename'], engine='python', skipfooter=0)
r_labels = pandas.read_csv(rimages_csv_file, usecols=['angle'], engine='python', skipfooter=0)

c_inputs = pandas.read_csv(cimages_csv_file, usecols=['filename'], engine='python', skipfooter=0)
c_labels = pandas.read_csv(cimages_csv_file, usecols=['angle'], engine='python', skipfooter=0)

if not((len(m_inputs.values) and len(m_labels.values)) or \
       (len(l_inputs.values) and len(l_labels.values)) or \
       (len(r_inputs.values) and len(r_labels.values)) or \
       (len(c_inputs.values) and len(c_labels.values))):
    print "error in dataset"

# split into train and test datas
mtrain_x = m_inputs[:int(len(m_inputs.values)*train_size)]
mtrain_y = m_labels[:int(len(m_labels.values)*train_size)]
mval_x = m_inputs[-int(len(m_inputs.values)*val_size):]
mval_y = m_labels[-int(len(m_labels.values)*val_size):]

ltrain_x = l_inputs[:int(len(l_inputs.values)*train_size)]
ltrain_y = l_labels[:int(len(l_labels.values)*train_size)]
lval_x = l_inputs[-int(len(l_inputs.values)*val_size):]
lval_y = l_labels[-int(len(l_labels.values)*val_size):]

rtrain_x = r_inputs[:int(len(r_inputs.values)*train_size)]
rtrain_y = r_labels[:int(len(r_labels.values)*train_size)]
rval_x = r_inputs[-int(len(r_inputs.values)*val_size):]
rval_y = r_labels[-int(len(r_labels.values)*val_size):]

ctrain_x = c_inputs[:int(len(c_inputs.values)*train_size)]
ctrain_y = c_labels[:int(len(c_labels.values)*train_size)]
cval_x = c_inputs[-int(len(c_inputs.values)*val_size):]
cval_y = c_labels[-int(len(c_labels.values)*val_size):]

#shuffle the data
# c = list(zip(ctrain_x, ctrain_y))
# random.shuffle(c)
# ctrain_x[:], ctrain_y[:] = zip(*c)
#
# c = list(zip(cval_x, cval_y))
# random.shuffle(c)
# cval_x[:], cval_y[:] = zip(*c)

# get the length
mlen_train = len(mtrain_x)
mlen_val = len(mval_x)

llen_train = len(ltrain_x)
llen_val = len(lval_x)

rlen_train = len(rtrain_x)
rlen_val = len(rval_x)

clen_train = len(ctrain_x)
clen_val = len(cval_x)

def loadY(str1, str2):
    if str1 == "merged" and str2 == "train":
        trainy = (ltrain_y.values[:, 0] + rtrain_y.values[:, 0] + ctrain_y.values[:, 0])
        trainy = trainy/3.0
    elif str1 == "left" and str2 == "train":
        trainy = ltrain_y.values[:, 0]
    elif str1 == "right" and str2 == "train":
        trainy = rtrain_y.values[:, 0]
    elif str1 == "center" and str2 == "train":
        trainy = ctrain_y.values[:, 0]
    elif str1 == "merged" and str2 == "validate":
        trainy = (lval_y.values[:, 0] + rval_y.values[:, 0] + cval_y.values[:, 0])
        trainy = trainy/3.0
    elif str1 == "left" and str2 == "validate":
        trainy = lval_y.values[:, 0]
    elif str1 == "right" and str2 == "validate":
        trainy = rval_y.values[:, 0]
    elif str1 == "center" and str2 == "validate":
        trainy = cval_y.values[:, 0]
    else:
        return

    trainy = numpy.expand_dims(trainy, axis = 1)
    return trainy*180/numpy.pi

def loadTrainDataM(batch_size):
    global mtrain_batch_index
    mtrain_x = []
    mtrain_y = []
    # fetch all the images and the labels
    for i in range(0,batch_size):
        img_file=os.path.join(data_dir, m_inputs.values[(mtrain_batch_index + i) % mlen_train][0][3:])
        x = cv2.imread(img_file)
        # normalise the image
        xt = cv2.resize(x.copy()/255.0, (640, 480)).astype(numpy.float32)
        xt = xt.transpose((2, 0, 1))
        mtrain_x.append(xt)
        # as the steering wheel angle is proportional to inverse of turning radius
        # we directly use the steering wheel angle (source: NVIDIA uses the inverse of turning radius)
        # but converted to radians

        mtrain_y.append(m_labels.values[(mtrain_batch_index + i) % mlen_train][0])
    mtrain_x = numpy.array(mtrain_x)
    mtrain_y = numpy.expand_dims(mtrain_y, axis = 1)
    #increment the index
    mtrain_batch_index += batch_size

    return mtrain_x, mtrain_y

def loadTrainDataL(batch_size):
    global ltrain_batch_index
    ltrain_x = []
    ltrain_y = []
    # fetch all the images and the labels
    for i in range(0,batch_size):
        img_file=os.path.join(data_dir, l_inputs.values[(ltrain_batch_index + i) % llen_train][0][3:])
        x = cv2.imread(img_file)
        # normalise the image
        xt = cv2.resize(x.copy()/255.0, (640, 480)).astype(numpy.float32)
        xt = xt.transpose((2, 0, 1))
        ltrain_x.append(xt)

        # as the steering wheel angle is proportional to inverse of turning radius
        # we directly use the steering wheel angle (source: NVIDIA uses the inverse of turning radius)
        # but converted to radians
        ltrain_y.append(l_labels.values[(ltrain_batch_index + i) % llen_train][0])
    ltrain_x = numpy.array(ltrain_x)
    ltrain_y = numpy.expand_dims(ltrain_y, axis = 1)
    #increment the index
    ltrain_batch_index += batch_size

    return ltrain_x, ltrain_y

def loadTrainDataR(batch_size):
    global rtrain_batch_index
    rtrain_x = []
    rtrain_y = []
    # fetch all the images and the labels
    for i in range(0,batch_size):
        img_file=os.path.join(data_dir, r_inputs.values[(rtrain_batch_index + i) % rlen_train][0][3:])
        x = cv2.imread(img_file)
        # normalise the image
        xt = cv2.resize(x.copy()/255.0, (640, 480)).astype(numpy.float32)
        xt = xt.transpose((2, 0, 1))
        rtrain_x.append(xt)

        # as the steering wheel angle is proportional to inverse of turning radius
        # we directly use the steering wheel angle (source: NVIDIA uses the inverse of turning radius)
        # but converted to radians
        rtrain_y.append(r_labels.values[(rtrain_batch_index + i) % rlen_train][0])
    rtrain_x = numpy.array(rtrain_x)
    rtrain_y = numpy.expand_dims(rtrain_y, axis = 1)
    #increment the index
    rtrain_batch_index += batch_size

    return rtrain_x, rtrain_y

def loadTrainDataC(batch_size):
    global ctrain_batch_index
    ctrain_x = []
    ctrain_y = []
    # fetch all the images and the labels
    for i in range(0,batch_size):
        img_file=os.path.join(data_dir, c_inputs.values[(ctrain_batch_index + i) % clen_train][0][3:])
        x = cv2.imread(img_file)
        # normalise the image
        xt = cv2.resize(x.copy()/255.0, (640, 480)).astype(numpy.float32)
        xt = xt.transpose((2, 0, 1))
        ctrain_x.append(xt)

        # as the steering wheel angle is proportional to inverse of turning radius
        # we directly use the steering wheel angle (source: NVIDIA uses the inverse of turning radius)
        # but converted to radians
        ctrain_y.append(c_labels.values[(ctrain_batch_index + i) % clen_train][0])
    ctrain_x = numpy.array(ctrain_x)
    ctrain_y = numpy.expand_dims(ctrain_y, axis = 1)
    #increment the index
    ctrain_batch_index += batch_size

    return ctrain_x, ctrain_y

def loadValDataM(batch_size):
    global mval_batch_index
    mval_x = []
    mval_y = []
    # fetch all the images and the labels
    for i in range(0,batch_size):
        img_file=os.path.join(data_dir, m_inputs.values[(mval_batch_index + i) % mlen_val][0][3:])
        x = cv2.imread(img_file)
        # normalise the image
        xt = cv2.resize(x.copy()/255.0, (640, 480)).astype(numpy.float32)
        xt = xt.transpose((2, 0, 1))
        xt = numpy.expand_dims(xt, axis = 0)
        mval_x.append(xt)

        # as the steering wheel angle is proportional to inverse of turning radius
        # we directly use the steering wheel angle (source: NVIDIA uses the inverse of turning radius)
        # but converted to radians
        mval_y.append(m_labels.values[(mval_batch_index + i) % mlen_val][0])

    mval_x = numpy.array(mval_x)
    mval_y = numpy.array(mval_y)
    #increment the index
    mval_batch_index += batch_size

    return mval_x, mval_y

def loadValDataL(batch_size):
    global lval_batch_index
    lval_x = []
    lval_y = []
    # fetch all the images and the labels
    for i in range(0,batch_size):
        img_file=os.path.join(data_dir, l_inputs.values[(lval_batch_index + i) % llen_val][0][3:])
        x = cv2.imread(img_file)
        # normalise the image
        xt = cv2.resize(x.copy()/255.0, (640, 480)).astype(numpy.float32)
        xt = xt.transpose((2, 0, 1))
        xt = numpy.expand_dims(xt, axis = 0)
        lval_x.append(xt)

        # as the steering wheel angle is proportional to inverse of turning radius
        # we directly use the steering wheel angle (source: NVIDIA uses the inverse of turning radius)
        # but converted to radians
        lval_y.append(l_labels.values[(lval_batch_index + i) % llen_val][0])

    lval_x = numpy.array(lval_x)
    lval_y = numpy.array(lval_y)
    #increment the index
    lval_batch_index += batch_size

    return lval_x, lval_y

def loadValDataR(batch_size):
    global rval_batch_index
    rval_x = []
    rval_y = []
    # fetch all the images and the labels
    for i in range(0,batch_size):
        img_file=os.path.join(data_dir, r_inputs.values[(rval_batch_index + i) % rlen_val][0][3:])
        x = cv2.imread(img_file)
        # normalise the image
        xt = cv2.resize(x.copy()/255.0, (640, 480)).astype(numpy.float32)
        xt = xt.transpose((2, 0, 1))
        xt = numpy.expand_dims(xt, axis = 0)
        rval_x.append(xt)

        # as the steering wheel angle is proportional to inverse of turning radius
        # we directly use the steering wheel angle (source: NVIDIA uses the inverse of turning radius)
        # but converted to radians
        rval_y.append(r_labels.values[(rval_batch_index + i) % rlen_val][0])

    rval_x = numpy.array(rval_x)
    rval_y = numpy.array(rval_y)
    #increment the index
    rval_batch_index += batch_size

    return rval_x, rval_y

def loadValDataC(batch_size):
    global cval_batch_index
    cval_x = []
    cval_y = []
    # fetch all the images and the labels
    for i in range(0,batch_size):
        img_file=os.path.join(data_dir, c_inputs.values[(cval_batch_index + i) % clen_val][0][3:])
        x = cv2.imread(img_file)
        # normalise the image
        xt = cv2.resize(x.copy()/255.0, (640, 480)).astype(numpy.float32)
        xt = xt.transpose((2, 0, 1))
        xt = numpy.expand_dims(xt, axis = 0)
        cval_x.append(xt)

        # as the steering wheel angle is proportional to inverse of turning radius
        # we directly use the steering wheel angle (source: NVIDIA uses the inverse of turning radius)
        # but converted to radians
        cval_y.append(c_labels.values[(cval_batch_index + i) % clen_val][0])

    cval_x = numpy.array(cval_x)
    cval_y = numpy.array(cval_y)
    #increment the index
    cval_batch_index += batch_size

    return cval_x, cval_y

def trainCDataGenerator():
    while 1:
        for i in range(0,clen_train):
            img_file=os.path.join(data_dir, c_inputs.values[i][0][3:])
            x = cv2.imread(img_file)
            # normalise the image
            xt = cv2.resize(x.copy()/255.0, (160, 120)).astype(numpy.float32)
            xt = xt.transpose((2, 0, 1))
            xt = numpy.expand_dims(xt, axis = 0)
            # as the steering wheel angle is proportional to inverse of turning radius
            # we directly use the steering wheel angle (source: NVIDIA uses the inverse of turning radius)
            # but converted to radians
            yt = numpy.expand_dims(c_labels.values[i][0], axis = 1)
            yield (numpy.array(xt), numpy.array(yt))

def valCDataGenerator():
    while 1:
        for i in range(0,clen_train):
            img_file=os.path.join(data_dir, c_inputs.values[i][0][3:])
            x = cv2.imread(img_file)
            # normalise the image
            xt = cv2.resize(x.copy()/255.0, (160, 120)).astype(numpy.float32)
            xt = xt.transpose((2, 0, 1))
            xt = numpy.expand_dims(xt, axis = 0)
            yield (numpy.array(xt))

def trainDataGen(str):
    global ltrain_batch_index
    global rtrain_batch_index
    global ctrain_batch_index
    while 1:
        train_x = []
        train_y = []
        # fetch all the images and the labels
        for i in range(0,getTrainBatchSize(str)):
            if str == 'left':
                img_file=os.path.join(data_dir, ltrain_x.values[(ltrain_batch_index + i) % llen_train][0][3:])
                yt = ltrain_y.values[(ltrain_batch_index + i) % llen_train][0]
            elif str == 'right':
                img_file=os.path.join(data_dir, rtrain_x.values[(rtrain_batch_index + i) % rlen_train][0][3:])
                yt = rtrain_y.values[(rtrain_batch_index + i) % rlen_train][0]
            elif str == 'center':
                img_file=os.path.join(data_dir, ctrain_x.values[(ctrain_batch_index + i) % clen_train][0][3:])
                yt = ctrain_y.values[(ctrain_batch_index + i) % clen_train][0]
            else:
                print 'error in string'

            x = cv2.imread(img_file)
            # normalise the image
            xt = cv2.resize(x.copy()/255.0, (160,120)).astype(numpy.float32)
            xt = xt.transpose((2, 0, 1))
            train_x.append(xt)

            # as the steering wheel angle is proportional to inverse of turning radius
            # we directly use the steering wheel angle (source: NVIDIA uses the inverse of turning radius)
            # but converted to radians
            train_y.append(yt)
        train_x = numpy.array(train_x)
        train_y = numpy.expand_dims(train_y, axis = 1)
        incTrainIndex(str)
        yield (train_x, train_y*180/numpy.pi)

def incTrainIndex(str):
    global ltrain_batch_index
    global rtrain_batch_index
    global ctrain_batch_index

    if str == 'left':
        #increment the index
        ltrain_batch_index += getTrainBatchSize(str)
    elif str == 'right':
        #increment the index
        rtrain_batch_index += getTrainBatchSize(str)
    elif str == 'center':
        #increment the index
        ctrain_batch_index += getTrainBatchSize(str)
    else:
        print 'error in string'

def valDataGen(str):
    global lval_batch_index
    global rval_batch_index
    global cval_batch_index

    while 1:
        val_x = []
        val_y = []
        # fetch all the images and the labels
        for i in range(0,getValBatchSize(str)):
            if str == 'left':
                img_file=os.path.join(data_dir, lval_x.values[(lval_batch_index + i) % llen_val][0][3:])
                yt = lval_y.values[(lval_batch_index + i) % llen_val][0]
            elif str == 'right':
                img_file=os.path.join(data_dir, rval_x.values[(rval_batch_index + i) % rlen_val][0][3:])
                yt = rval_y.values[(rval_batch_index + i) % rlen_val][0]
            elif str == 'center':
                img_file=os.path.join(data_dir, cval_x.values[(cval_batch_index + i) % clen_val][0][3:])
                yt = cval_y.values[(cval_batch_index + i) % clen_val][0]
            else:
                print 'error in string'

            x = cv2.imread(img_file)
            # normalise the image
            xt = cv2.resize(x.copy()/255.0, (160,120)).astype(numpy.float32)
            xt = xt.transpose((2, 0, 1))
            val_x.append(xt)

            # as the steering wheel angle is proportional to inverse of turning radius
            # we directly use the steering wheel angle (source: NVIDIA uses the inverse of turning radius)
            # but converted to radians
            val_y.append(yt)
        val_x = numpy.array(val_x)
        val_y = numpy.expand_dims(val_y, axis = 1)
        incValIndex(str)
        yield (val_x, val_y*180/numpy.pi)

def incValIndex(str):
    global lval_batch_index
    global rval_batch_index
    global cval_batch_index

    if str == 'left':
        #increment the index
        lval_batch_index += getValBatchSize(str)
    elif str == 'right':
        #increment the index
        rval_batch_index += getValBatchSize(str)
    elif str == 'center':
        #increment the index
        cval_batch_index += getValBatchSize(str)
    else:
        print 'error in string'

def getValBatchSize(str):
    global lval_batch_index
    global rval_batch_index
    global cval_batch_index

    if str == 'left' and (llen_val - (lval_batch_index%llen_val)) < batch_size:
        size = llen_val - (lval_batch_index%llen_val)
    elif str == 'right' and (rlen_val - (rval_batch_index%rlen_val)) < batch_size:
        size = rlen_val - (rval_batch_index%rlen_val)
    elif str == 'center' and (clen_val - (cval_batch_index%clen_val)) < batch_size:
        size = clen_val - (cval_batch_index%clen_val)
    else:
        size = batch_size

    return size

def getTrainBatchSize(str):
    global ltrain_batch_index
    global rtrain_batch_index
    global ctrain_batch_index

    if str == 'left' and (llen_train - (ltrain_batch_index%llen_train)) < batch_size:
        size = llen_train - (ltrain_batch_index%llen_train)
    elif str == 'right' and (rlen_train - (rtrain_batch_index%rlen_train)) < batch_size:
        size = rlen_train - (rtrain_batch_index%rlen_train)
    elif str == 'center' and (clen_train - (ctrain_batch_index%clen_train)) < batch_size:
        size = clen_train - (ctrain_batch_index%clen_train)
    else:
        size = batch_size

    return size

### as of now using the validation data for testing as well
def testDataGen(str):
    global ltest_batch_index
    global rtest_batch_index
    global ctest_batch_index

    while 1:
        val_x = []
        # fetch all the images and the labels
        for i in range(0,getTestBatchSize(str)):
            if str == 'left':
                img_file=os.path.join(data_dir, lval_x.values[(ltest_batch_index + i) % llen_val][0][3:])
            elif str == 'right':
                img_file=os.path.join(data_dir, rval_x.values[(rtest_batch_index + i) % rlen_val][0][3:])
            elif str == 'center':
                img_file=os.path.join(data_dir, cval_x.values[(ctest_batch_index + i) % clen_val][0][3:])
            else:
                print 'error in string'

            x = cv2.imread(img_file)
            # normalise the image
            xt = cv2.resize(x.copy()/255.0, (160,120)).astype(numpy.float32)
            xt = xt.transpose((2, 0, 1))
            val_x.append(xt)
        val_x = numpy.array(val_x)
        incTestIndex(str)
        yield (val_x)

def incTestIndex(str):
    global ltest_batch_index
    global rtest_batch_index
    global ctest_batch_index

    if str == 'left':
        #increment the index
        ltest_batch_index += getTestBatchSize(str)
    elif str == 'right':
        #increment the index
        rtest_batch_index += getTestBatchSize(str)
    elif str == 'center':
        #increment the index
        ctest_batch_index += getTestBatchSize(str)
    else:
        print 'error in string'

def getTestBatchSize(str):
    global ltest_batch_index
    global rtest_batch_index
    global ctest_batch_index

    if str == 'left' and (llen_val - (ltest_batch_index%llen_val)) < batch_size:
        size = llen_val - (ltest_batch_index%llen_val)
    elif str == 'right' and (rlen_val - (rtest_batch_index%rlen_val)) < batch_size:
        size = rlen_val - (rtest_batch_index%rlen_val)
    if str == 'center' and (clen_val - (ctest_batch_index%clen_val)) < batch_size:
        size = clen_val - (ctest_batch_index%clen_val)
    else:
        size = batch_size

    return size

############# merged model generators

## train generator

def getMTrainBatchSize():
    global mdata_batch_index

    if (clen_train - (mdata_batch_index%clen_train)) < batch_size:
        size = clen_train - (mdata_batch_index%clen_train)
    else:
        size = batch_size

    return size

def incMTrainIndex():
    global mdata_batch_index
    mdata_batch_index += getMTrainBatchSize()

def trainMDataGen():
    global mdata_batch_index

    while 1:
        train_lx = []
        train_rx = []
        train_cx = []
        train_y = []
        # fetch all the images and the labels
        for i in range(0,getMTrainBatchSize()):
            img_file=os.path.join(data_dir, ltrain_x.values[(mdata_batch_index + i) % llen_train][0][3:])
            x = cv2.imread(img_file)
            # normalise the image
            xt = cv2.resize(x.copy()/255.0, (160,120)).astype(numpy.float32)
            xt = xt.transpose((2, 0, 1))
            train_lx.append(xt)
            lyt = ltrain_y.values[(mdata_batch_index + i) % llen_train][0]

            img_file=os.path.join(data_dir, rtrain_x.values[(mdata_batch_index + i) % rlen_train][0][3:])
            x = cv2.imread(img_file)
            # normalise the image
            xt = cv2.resize(x.copy()/255.0, (160,120)).astype(numpy.float32)
            xt = xt.transpose((2, 0, 1))
            train_rx.append(xt)
            ryt = rtrain_y.values[(mdata_batch_index + i) % rlen_train][0]

            img_file=os.path.join(data_dir, ctrain_x.values[(mdata_batch_index + i) % clen_train][0][3:])
            yt = ctrain_y.values[(mdata_batch_index + i) % clen_train][0]
            x = cv2.imread(img_file)
            # normalise the image
            xt = cv2.resize(x.copy()/255.0, (160,120)).astype(numpy.float32)
            xt = xt.transpose((2, 0, 1))
            train_cx.append(xt)
            cyt = rtrain_y.values[(mdata_batch_index + i) % clen_train][0]
            yt = (lyt + ryt + cyt)/3.0

            # as the steering wheel angle is proportional to inverse of turning radius
            # we directly use the steering wheel angle (source: NVIDIA uses the inverse of turning radius)
            # but converted to radians
            train_y.append(yt)
        train_y = numpy.expand_dims(train_y, axis = 1)
        incMTrainIndex()
        yield [numpy.array(train_lx), numpy.array(train_rx), numpy.array(train_cx)], train_y*180/numpy.pi

## validation generator

def getMValBatchSize():
    global mrval_batch_index

    if (clen_val - (mrval_batch_index%clen_val)) < batch_size:
        size = clen_val - (mrval_batch_index%clen_val)
    else:
        size = batch_size

    return size

def incMValIndex():
    global mrval_batch_index
    mrval_batch_index += getMValBatchSize()

def valMDataGen():
    global mrval_batch_index
    val_lx = []
    val_rx = []
    val_cx = []
    val_y = []

    while 1:
        # we will only take center images as of now
        for i in range(0,getMValBatchSize()):
            img_file=os.path.join(data_dir, lval_x.values[(mrval_batch_index + i) % llen_val][0][3:])
            x = cv2.imread(img_file)
            # normalise the image
            xt = cv2.resize(x.copy()/255.0, (160,120)).astype(numpy.float32)
            xt = xt.transpose((2, 0, 1))
            val_lx.append(xt)
            lyt = lval_y.values[(mrval_batch_index + i) % llen_val][0]

            img_file=os.path.join(data_dir, rval_x.values[(mrval_batch_index + i) % rlen_val][0][3:])
            x = cv2.imread(img_file)
            # normalise the image
            xt = cv2.resize(x.copy()/255.0, (160,120)).astype(numpy.float32)
            xt = xt.transpose((2, 0, 1))
            val_rx.append(xt)
            ryt = rval_y.values[(mrval_batch_index + i) % rlen_val][0]

            img_file=os.path.join(data_dir, cval_x.values[(mrval_batch_index + i) % clen_val][0][3:])
            yt = cval_y.values[(mrval_batch_index + i) % clen_val][0]
            x = cv2.imread(img_file)
            # normalise the image
            xt = cv2.resize(x.copy()/255.0, (160,120)).astype(numpy.float32)
            xt = xt.transpose((2, 0, 1))
            val_cx.append(xt)
            cyt = rval_y.values[(mrval_batch_index + i) % clen_val][0]

            yt = cyt; #(lyt + ryt + cyt)/3.0

            # as the steering wheel angle is proportional to inverse of turning radius
            # we directly use the steering wheel angle (source: NVIDIA uses the inverse of turning radius)
            # but converted to radians
            val_y.append(yt)
        val_y = numpy.expand_dims(val_y, axis = 1)
        incMvalIndex()
        yield (numpy.array(val_cx), val_y*180/numpy.pi) #numpy.array(val_rx), numpy.array(val_cx), val_y*180/numpy.pi)

## test generator

def getMtestBatchSize():
    global mtest_batch_index

    if (clen_test - (mtest_batch_index%clen_test)) < batch_size:
        size = clen_test - (mtest_batch_index%clen_test)
    else:
        size = batch_size

    return size

def incMtestIndex():
    global mtest_batch_index
    mtest_batch_index += getMtestBatchSize()

def testDataGen():
    global mtest_batch_index
    test_lx = []
    test_rx = []
    test_cx = []

    while 1:
        # we will only take center images as of now
        for i in range(0,getMtestBatchSize):
            img_file=os.path.join(data_dir, ltest_x.testues[(mtest_batch_index + i) % llen_test][0][3:])
            x = cv2.imread(img_file)
            # normalise the image
            xt = cv2.resize(x.copy()/255.0, (160,120)).astype(numpy.float32)
            xt = xt.transpose((2, 0, 1))
            test_lx.append(xt)

            img_file=os.path.join(data_dir, rtest_x.testues[(mtest_batch_index + i) % rlen_test][0][3:])
            x = cv2.imread(img_file)
            # normalise the image
            xt = cv2.resize(x.copy()/255.0, (160,120)).astype(numpy.float32)
            xt = xt.transpose((2, 0, 1))
            test_rx.append(xt)

            img_file=os.path.join(data_dir, ctest_x.testues[(mtest_batch_index + i) % clen_test][0][3:])
            yt = ctest_y.testues[(mtest_batch_index + i) % clen_test][0]
            x = cv2.imread(img_file)
            # normalise the image
            xt = cv2.resize(x.copy()/255.0, (160,120)).astype(numpy.float32)
            xt = xt.transpose((2, 0, 1))
            test_cx.append(xt)

        incMtestIndex()
        yield (numpy.array(test_cx)) #, numpy.array(test_rx), numpy.array(test_cx))
