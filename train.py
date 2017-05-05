import csv
import os
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

def read_csv_samples(samples, path, skip = True):
    with open (os.path.join(path, 'driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for sample in reader:
            if skip == True:
                skip = False
            else:
                sample[0] = os.path.join(path, sample[0].strip())
                sample[1] = os.path.join(path, sample[1].strip())
                sample[2] = os.path.join(path, sample[2].strip())
                if os.path.exists(sample[0]) == False or os.path.exists(sample[1]) == False or os.path.exists(sample[2]) == False:
                    print(sample[0], sample[1], sample[2], "not exist!")
                    break
                samples.append(sample)
    return samples

samples = []
samples = read_csv_samples(samples, "data")

MYDATA="mydata"
if os.path.isdir(MYDATA) and os.path.exists(MYDATA):
    print("Dir " + MYDATA + " exists")
    for f in os.listdir(MYDATA):
        f_root = os.path.join(MYDATA, f, "data")
        if os.path.isdir(f_root):
            print("read data from", f_root)
            samples = read_csv_samples(samples, f_root, skip=False)

print("samples ", len(samples))
np.random.shuffle(samples)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    sample_size = int(batch_size / 2)
    while 1: # Loop forever so the generator never terminates
        np.random.shuffle(samples)
        for offset in range(0, num_samples, sample_size):
            batch_samples = samples[offset:offset+sample_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                #name = './IMG/'+batch_sample[0].split('/')[-1]
                name = batch_sample[0]
                if os.path.exists(name) != True:
                    print(name, "not exist!")
                    break
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                images.append(cv2.flip(center_image, 1))
                angles.append(-center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

'''
X_train = np.array(images)
y_train = np.array(measurements)

print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
'''

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
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
model.fit_generator(train_generator, samples_per_epoch= \
                    len(train_samples) * 2, validation_data=validation_generator, \
                                nb_val_samples=len(validation_samples) * 2, nb_epoch=5)

model.save('model.h5')

