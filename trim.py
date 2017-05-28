import csv
import copy
import os
import cv2
import numpy as np

samples = []
with open ('select.csv') as csvfile:
    reader = csv.reader(csvfile)
    for sample in reader:
        if os.path.exists(sample[0]) == False:
            print(sample[0], "not exist!")
        else:
            #print(sample[0], "push back")
            samples.append(sample)


with open ('select2.csv', 'w') as ofile:
    writer = csv.writer(ofile)
    for sample in samples:
        writer.writerow(sample)

