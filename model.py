import csv
import os
import cv2
import numpy as np
import sklearn
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# read and parse csv file
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

# read and parse csv file
def read_samples(samples, path, skip = False):
    with open (os.path.join(path, 'gen.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for sample in reader:
            if skip == True:
                skip = False
            else:
                #print("read ", sample[0])
                sample[0] = os.path.join(path, sample[0].strip())
                sample[1] = os.path.join(path, sample[1].strip())
                sample[2] = os.path.join(path, sample[2].strip())
                if os.path.exists(sample[0]) == False:
                    print(sample[0], "not exist!")
                    #break
                    continue
                samples.append(sample)
    return samples

samples = []

# read udacity's data
#samples = read_csv_samples(samples, "data")

# read my recorded data
MYDATA="mydata"
if os.path.isdir(MYDATA) and os.path.exists(MYDATA):
    print("Dir " + MYDATA + " exists")
    for f in os.listdir(MYDATA):
        f_root = os.path.join(MYDATA, f, "data")
        if os.path.isdir(f_root):
            print("read data from", f_root)
            #samples = read_csv_samples(samples, f_root, skip=False)

samples = read_samples(samples, "./")
#samples = read_samples(samples, "./")

print("samples ", len(samples))
np.random.shuffle(samples)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def normalize_gray(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = np.float64(img)
    return img / 255.0 - 0.5

def generator(samples, batch_size=32):
    num_samples = len(samples)
    sample_size = batch_size
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
                center_image = normalize_gray(center_image)
                center_image = np.expand_dims(center_image, axis=2)
                #print("shape ", center_image.shape)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                #images.append(cv2.flip(center_image, 1))
                #angles.append(-center_angle)

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
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D
from keras.layers import Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Cropping2D(cropping=((56,20), (0,0)), input_shape=(160,320,1)))
#model.add(Lambda(lambda x: x / 255.0 - 0.5))
# conv layer 1
model.add(Convolution2D(24,5,5,activation="relu"))
model.add(MaxPooling2D(border_mode='same'))
# conv layer 2
model.add(Convolution2D(36,5,5,activation="relu"))
model.add(MaxPooling2D(border_mode='same'))
# conv layer 3
model.add(Convolution2D(48,5,5,activation="relu"))
model.add(MaxPooling2D(border_mode='same'))
# conv layer 4
model.add(Convolution2D(64,3,3,activation="relu"))
#model.add(MaxPooling2D(border_mode='same'))
# conv layer 5
model.add(Convolution2D(128,3,3,activation="relu"))
#model.add(MaxPooling2D(border_mode='same'))
# conv layer 6
model.add(Convolution2D(160,3,3,activation="relu"))
#model.add(MaxPooling2D(border_mode='same'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024))
model.add(Dense(512))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
model.summary()

history = model.fit_generator(train_generator, samples_per_epoch= \
                    len(train_samples), validation_data=validation_generator, \
                                nb_val_samples=len(validation_samples), nb_epoch=1)

# save model
print("Saving model")
model.save('model.h5')
with open('model.json', 'w') as outfile:
	json.dump(model.to_json(), outfile)

### print the keys contained in the history object
print(history.history)

### plot the training and validation loss for each epoch
'''
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('loss.png')
'''
