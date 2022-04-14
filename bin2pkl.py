import struct
import numpy as np
import pickle
import argparse


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", help='sfbay_l, sfbay_c, flevo_l, or flevo_c', required=True)
ap.add_argument("-c", "--channels", help='number of channels.', required=True)
ap.add_argument("-p", "--patch", help='sliding window patch size.', required=True)
ap.add_argument("-W", help='width size of the scene.', required=True)
ap.add_argument("-H", help='height of the scene.', required=True)
args = vars(ap.parse_args())
channels = int(args["channels"])
patch_size = int(args["patch"])
m_W = int(args["W"])
m_H = int(args["H"])

filesize = m_W * m_H

channels = 6

def read_row(filename):
    rdata = np.zeros(filesize, dtype = 'float32')
    f = open(filename, 'rb')

    for l in range(0, +filesize):
        (num,) = struct.unpack('f', f.read(4))
        rdata[l] = num
    return rdata

row_data = np.zeros((filesize, channels), dtype = 'float32')
if channels == 3:
    filenames = ['T11dbs.bin', 'T22dbs.bin', 'T33dbs.bin']
elif channels == 4:
    filenames = ['T11dbs.bin', 'T22dbs.bin', 'T33dbs.bin', 'span_dbs.bin']
elif channels == 6:
    filenames = ['T11dbs.bin', 'T22dbs.bin', 'T33dbs.bin', 'C11dbs.bin', 'C22dbs.bin', 'C33dbs.bin']

for i in range(0, channels):
    row_data[:, i] = read_row(args["dataset"] + '/' + filenames[i])

# Saving the objects:
with open(args["dataset"] + '/' + args["dataset"] + '_' + str(channels) + '_rowdata.pkl', 'wb') as f:
    pickle.dump([row_data], f, protocol = 4)


import numpy as np
import struct
import pickle

if args["dataset"] == 'sfbay_l':
    train_size = 1462 # SFBay_L
    test_size = 121997 # SFBay_L
elif args["dataset"] == 'sfbay_c':
    train_size = 2500 # SFBay_C
    test_size = 250000 # SFBay_C
elif args["dataset"] == 'flevo_l':
    #train_size = 964 # Flevo_L 12 class
    #test_size = 169757 # Flevo_L 12 class
    train_size = 4211 # Flevo_L
    test_size = 204023 # Flevo_L
elif args["dataset"] == 'flevo_c':
    train_size = 2000 # Flevo_C
    test_size = 200000 # Flevo_C

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
train_positions, train_labels = read_gtd(args["dataset"] + '/' + args["dataset"] + '_train', train_size)
test_positions, test_labels = read_gtd(args["dataset"] + '/' + args["dataset"] + '_test', test_size)
#Converting from float to int
train_positions = np.array(train_positions, dtype = 'int')
test_positions = np.array(test_positions, dtype = 'int')
train_labels = np.array(train_labels, dtype = 'int')
test_labels = np.array(test_labels, dtype = 'int')

#dir + title + '_' + str(channels) + '_rowdata.pkl'
with open(args["dataset"] + '/' + args["dataset"] + '_' + str(channels) + '_rowdata.pkl', 'rb') as f:
    data = pickle.load(f)

data = np.array(data[0])
data = np.reshape(data, [m_H, m_W, channels])

data_padded = np.zeros([m_H + patch_size - 1, m_W + patch_size - 1, channels], dtype='float32')
startid = (patch_size - 1) // 2
data_padded[startid:startid + m_H, startid:startid + m_W, :] = data


patch_data = np.zeros([filesize, patch_size, patch_size, channels], dtype = 'float32')

index = 0
for i in range(startid, startid + m_H):
    for j in range(startid, startid + m_W):
        for c in range(0, channels):
            patch_data[index, :, :, c] = data_padded[i - startid : i + startid + 1, j - startid : j + startid + 1, c]
        index = index + 1
        print('Processed: ', (index/filesize)*100)


train_data = np.zeros([train_size, patch_size, patch_size, channels], dtype = 'float32')
test_data = np.zeros([test_size, patch_size, patch_size, channels], dtype = 'float32')
train_data = patch_data[train_positions]
test_data = patch_data[test_positions]


with open(args["dataset"] + '/' + args["dataset"] + '_' + str(channels) + '_' + str(patch_size) +'.pkl', 'wb') as f:
    pickle.dump([train_data, train_labels, test_data, test_labels], f, protocol = 4)
