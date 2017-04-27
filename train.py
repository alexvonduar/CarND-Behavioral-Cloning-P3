import csv
import cv2
import numpy as np
import os

'''
lines = []

skip = True
with open ('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if skip == True:
            skip = False
        else:
            lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    image = cv2.flip(image, 1)
    images.append(image)
    measurements.append(-measurement)
'''

def read_data(images, measurements, path, skip = True):
    lines = []

    #skip = True
    with open (os.path.join(path, 'driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if skip == True:
                skip = False
            else:
                lines.append(line)

    #images = []
    #measurements = []
    for line in lines:
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = os.path.join(path, 'IMG', filename)
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)
        image = cv2.flip(image, 1)
        images.append(image)
        measurements.append(-measurement)
        #print("Read image: ", current_path, measurement)
    return images,measurements


MYDATA="mydata"

images = []
measurements = []

images,measurements = read_data(images, measurements, "data")

if os.path.isdir(MYDATA) and os.path.exists(MYDATA):
    print("Dir " + MYDATA + " exists")
    for f in os.listdir(MYDATA):
        f_root = os.path.join(MYDATA, f, "data")
        if os.path.isdir(f_root):
            print("read data from", f_root)
            read_data(images, measurements, f_root, skip=False)
        #else:
        #    print("file " + f_root)


X_train = np.array(images)
y_train = np.array(measurements)

print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D
from keras.layers import Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
# conv layer 1
model.add(Convolution2D(24,5,5,activation="relu"))
model.add(MaxPooling2D())
# conv layer 2
model.add(Convolution2D(36,5,5,activation="relu"))
model.add(MaxPooling2D())
# conv layer 3
model.add(Convolution2D(48,5,5,activation="relu"))
model.add(MaxPooling2D())
# conv layer 4
model.add(Convolution2D(64,3,3,activation="relu"))
#model.add(MaxPooling2D())
# conv layer 6
model.add(Convolution2D(64,3,3,activation="relu"))
#model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')
