import csv
import copy
import os
import cv2
import numpy as np

GEN_DIR="gendir"

with open ('gen.csv') as csvfile:
    reader = csv.reader(csvfile)
    for sample in reader:
        if os.path.exists(sample[0]) == False:
            print(sample[0], "not exist!")
            break
        else:
            print(sample[0])
