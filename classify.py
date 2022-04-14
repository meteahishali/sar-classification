import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint

from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
import pickle
import numpy as np
import math
import argparse

import tensorflow as tf
import random
np.random.seed(1)
tf.random.set_seed(1)
random.seed(1)

###### Get CNN model
def get_CNN(input_shape):
    
    input = Input(shape = input_shape, name='input1')

    # First branch
    x_1 = Conv2D(20, 3, strides=1, activation='tanh', padding = 'valid')(input)
    x_1 = MaxPooling2D(pool_size=x_1.shape[1])(x_1)
    x_1 = Flatten()(x_1)
    x_1 = Dense(10, activation="tanh")(x_1)
    y = Dense(N, activation="softmax")(x_1)
    
    CNNmodel = Model(inputs = input, outputs = y)

    return CNNmodel
################################# 

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", help='sfbay_l, sfbay_c, flevo_l, or flevo_c', required=True)
ap.add_argument("-c", "--channels", help='number of channels.', required=True)
ap.add_argument("-p", "--patch", help='sliding window patch size.', required=True)
args = vars(ap.parse_args())
channels = int(args["channels"])
patch_size = int(args["patch"])

weights = False
modelName = args["dataset"]+ '_patch_' + str(patch_size)


with open(args["dataset"]  + '/' + args["dataset"] + '_' + str(channels) + '_' + str(patch_size) +'.pkl', 'rb') as f:############
    x_train, y_train, x_test, y_test = pickle.load(f)

##################
print("Max and min values of data:")
print(x_test.max())
print(x_test.min())
print(x_train.max())
print(x_train.min())

N = len(np.unique(y_train))
print('Number of classes: ', N)

y_temp_train = np.array(y_train)
y_train = to_categorical(y_temp_train)

##################
input_shape = (x_train.shape[-3], x_train.shape[-2], x_train.shape[-1]) #######################################

model = get_CNN(input_shape)
model.summary()
model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

def step_decay(epoch):
   initial_lrate = 0.1
   drop = 0.94
   epochs_drop = 2.0
   lrate = initial_lrate * math.pow(drop,  math.floor((1+epoch)/epochs_drop))
   return lrate
lrate = LearningRateScheduler(step_decay)

class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='loss', value=0.01, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

mc = ModelCheckpoint(modelName + '.h5', monitor='accuracy', mode='max', save_best_only=True, verbose=1)
callbacks_list = [lrate, mc]

if weights:
    model.load_weights(modelName + '.h5')
else:

    history = model.fit(x_train, y_train, epochs = 100, batch_size = 1, shuffle = True, callbacks = callbacks_list)
    model.load_weights(modelName + '.h5')
    loss_history = history.history["loss"]
    numpy_loss_history = np.array(loss_history)
    np.savetxt(modelName + "_loss_history.txt", numpy_loss_history, delimiter=",")
    with open(modelName + '_loss.pkl', 'wb') as f:
        pickle.dump([numpy_loss_history], f, protocol = 4)

    plt.figure(0)
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(modelName + "_loss.png")

y_pred = model.predict(x_test)
#y_pred = model.predict([x_test])
y_pred = np.argmax(y_pred, axis = 1)
print("Test accuracy is: ", accuracy_score(y_pred, y_test))

C = confusion_matrix(y_test, y_pred)
print("Class Accuracies: ")
s = 0
for i in range(0, C.shape[0]):
    s = s + C[i,i]
    print(C[i,i]/np.sum(C[i, :]))
print("Accuracy: ", s / len(y_pred))
print(C)




'''
import hdf5storage
mat = hdf5storage.loadmat(dataDir + name + '.mat')
data = mat['fff']
data = np.array(data, dtype = 'float32')

mat2 = hdf5storage.loadmat(dataDir2 + name2 + '.mat')
data2 = mat2['fff']
data2 = np.array(data2, dtype = 'float32')

## Creating overall mask
x_test = []


# San Diego colors
colors = np.zeros([4, 3])
colors[0, :] = [0, 1, 254]
colors[1, :] = [0, 131, 71]
colors[2, :] = [0, 253, 255]
colors[3, :] = [0, 255, 0]

'''
#test_size = 121997 # SFBay_L
#test_size = 204023 # Flevo_L
#test_size = 116494 # sd_p
#m_W = 1024 # SFBay_L
#m_H = 900
#m_W = 1024 # Flevo_L
#m_H = 750
# SFBay_L colors
colors = np.zeros([5, 3])
colors[0, :] = [0, 1, 254]
colors[1, :] = [0, 131, 71]
colors[2, :] = [0, 253, 255]
colors[3, :] = [0, 255, 0]
colors[4, :] = [255, 126, 0]

# Flevo_L colors
colors = np.zeros([12, 3])
colors[0, :] = [0, 1, 254]
colors[1, :] = [0, 131, 71]
colors[2, :] = [0, 253, 255]
colors[3, :] = [0, 255, 0]
colors[4, :] = [255, 126, 0]
colors[5, :] = [180, 0, 255]
colors[6, :] = [251, 255, 7]
colors[7, :] = [91, 8, 227]
colors[8, :] = [253, 0, 0]
colors[9, :] = [172, 138, 78]
colors[10, :] = [255, 181, 230]
colors[11, :] = [191, 191, 255]
'''

#with open(dataDir + title + 'all_' + str(channels) + '_' + str(patch_size) +'.pkl', 'rb') as f:
#    x_all = pickle.load(f)
data = data[:, 0:22]
data2 = data2[:, 0:22]
data = np.expand_dims(data, axis=2)
data2 = np.expand_dims(data2, axis=2)
y_all = model.predict([data, data2])
y_all = np.argmax(y_all, axis = 1)
print(y_all)

color_mask = np.zeros([m_H, m_W, 3])
counter = 0
for i in range(0, m_H):
    for j in range(0, m_W):
        color_mask[i, j, :] = colors[y_all[counter], :]
        counter = counter + 1
import cv2
color_mask = color_mask.astype(np.uint8)
color_mask = cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB)
cv2.imwrite('outputs3/masks/' + title + '_' + str(x_train.shape[-2]) +'_cl_mask.png', color_mask)

import struct
def read_gtd(filename, size):
    positions = np.zeros((size,), dtype = 'float32')
    labels = np.zeros((size,), dtype = 'float32')
    f1 = open(filename + '_positions.gtd', 'rb')
    f2 = open(filename + '_labels.gtd', 'rb')
    for l in range(0, size):
        (num1,) = struct.unpack('f', f1.read(4))
        (num2,) = struct.unpack('f', f2.read(4))
        positions[l] = num1
        labels[l] = num2
    return positions, labels

### Read gtd

test_positions, test_labels = read_gtd(dataDir + title + '_test', test_size)
#Converting from float to int
test_positions = np.array(test_positions, dtype = 'int')
test_labels = np.array(test_labels, dtype = 'int')

color_mask_layout = np.zeros([m_H, m_W, 3])
counter = 0
for i in range(0, m_H):
    for j in range(0, m_W):
        if counter in test_positions: color_mask_layout[i, j, :] = colors[y_all[counter], :]
        counter = counter + 1
import cv2
color_mask_layout = color_mask_layout.astype(np.uint8)
color_mask_layout = cv2.cvtColor(color_mask_layout, cv2.COLOR_BGR2RGB)
cv2.imwrite('outputs3/masks/' + title + '_' + str(x_train.shape[-2]) +'_cl_overlaid.png', color_mask_layout)
'''







































'''
### Detect uncertain regions
y_all2 = clf.predict_proba(features_all)
y_all2 = y_all2.max(axis=1)
print(y_all2)
color_mask = np.zeros([m_H, m_W, 3])
counter = 0
for i in range(0, m_H):
    for j in range(0, m_W):
        if y_all2[counter] > 0.5: color_mask[i, j, :] = colors[y_all[counter], :]
        counter = counter + 1
import cv2
color_mask = color_mask.astype(np.uint8)
color_mask = cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB)
cv2.imwrite('outputs/masks/' + title + '_' + str(channels) + '_' + str(patch_size) + '_black.png', color_mask)
'''
'''
// 0x00BGR
// Flevoland by Yu
(*temp) = 0x00FE0100;	temp++;//	Water
(*temp) = 0x00478300;	temp++;//	Forest	
(*temp) = 0x00FFFD00;	temp++;//	Lucerne
(*temp) = 0x0000FF00;	temp++;//	Grass
(*temp) = 0x00007EFF;	temp++;//	Rapeseed
(*temp) = 0x00FF00B4;	temp++;//	Beet 
(*temp) = 0x0007FFFB;	temp++;//	Potatoes
(*temp) = 0x00E3085B;	temp++;//	Peas
(*temp) = 0x000000FD;	temp++;//	Stem_Beans	
(*temp) = 0x004E8AAC;	temp++;//	Bare_Soil
(*temp) = 0x00E6B5FF;	temp++;//	Wheat
(*temp) = 0x00FFBFBF;	temp++;//	Wheat2 
(*temp) = 0x00BEFFBD;	temp++;//	Wheat3

(*temp) = 0x0000007F;	temp++;//	Barley
(*temp) = 0x0095E2FA;	temp++;//	Building 
(*temp) = 0x00008000;	temp++;//	Roads
(*temp) = 0x00000000;	temp++;//	Black
(*temp) = 0x00000000;	temp++;//	Black
(*temp) = 0x00000000;	temp++;//	Black
(*temp) = 0x00000000;	temp++;//	Black
'''

'''
import multiprocessing
num_cores = multiprocessing.cpu_count() # This is for the parallel implementation of nearest search on CPU.
def gammaSearch(i):
    gamma = gammas[i]

    clf = SVC(random_state = Hyperparams.random_state, kernel = kernel, decision_function_shape = decision_function, gamma = gamma)
    clf.fit(features_train, y_temp_train)
    acc = clf.score(features_test, y_temp)
    return acc

text_file = open("svm.txt", "w")
for decision_function in Hyperparams.decision_function:

    for kernel in Hyperparams.kernel:
        text_file.write("Decision Function: " + decision_function + ", Kernel: " + kernel + "\n")

        gammas = np.linspace(0.001, 0.0001, num = Hyperparams.gammaNumber, endpoint = True)
        accuracies = np.zeros((Hyperparams.gammaNumber, ))
        
        pool = multiprocessing.Pool(processes = num_cores)
        accuracies = pool.map(gammaSearch, range(len(gammas)))
        pool.close()
        pool.join()
        for i in range(0, len(accuracies)):
            text_file.write("Acuracy: " + str(accuracies[i]) + ", Gamma: " + str(gammas[i]) + "\n")
        text_file.write("Maximum accracy: " + str(np.max(accuracies)) + ", Gamma value: " + str(gammas[np.argmax(accuracies)]) + "\n")
        text_file.write("\n")
        print("For DF = ", decision_function, ", and kernel = ", kernel, " Maximum accracy: ", np.max(accuracies), "with Gamma value: ", gammas[np.argmax(accuracies)])
text_file.close()
'''
