import csv
import copy
import os
import cv2
import numpy as np

def read_csv_samples(samples, path, skip = False):
    with open (os.path.join(path, 'select.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for sample in reader:
            if skip == True:
                skip = False
            else:
                samples.append(sample)
    return samples

samples = []

# read udacity's data
samples = read_csv_samples(samples, "./")


print("samples ", len(samples))

for sample in samples:
    if os.path.exists(sample[0]) == True:
        img = cv2.imread(sample[0])
        crop_img = img[55:140,:,:]
        cv2.imshow("cropped", crop_img)
        cv2.waitKey(16)
    #else:
        #print(sample[0], "doesn't exist");
