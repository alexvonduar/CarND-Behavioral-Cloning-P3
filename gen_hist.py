import csv
import copy
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_csv_samples(angles, path, skip = True, csv_name='driving_log.csv'):
    with open (os.path.join(path, csv_name)) as csvfile:
        reader = csv.reader(csvfile)
        for sample in reader:
            if skip == True:
                skip = False
            else:
                name = os.path.join(path, sample[0])
                if os.path.exists(name) == False:
                    print(name, "not exist!")
                    continue
                angle = float(sample[3])
                angles.append(angle)

    return angles

def read_aug_samples(angles, path, skip = True, csv_name='driving_log.csv'):
    with open (os.path.join(path, csv_name)) as csvfile:
        reader = csv.reader(csvfile)
        for sample in reader:
            if skip == True:
                skip = False
            else:
                name = os.path.join(path, sample[0])
                if os.path.exists(name) == False:
                    print(name, "not exist!")
                    #break
                    continue
                angle = float(sample[3])
                angles.append(angle)
                # gen flip
                angles.append(-angle)

                correction = 0.15 + (np.random.random() - 0.5) * 0.01
                aug_angle = angle + correction
                if aug_angle > 1.0:
                    aug_angle = 1.0
                angles.append(aug_angle)

                correction = 0.15 + (np.random.random() - 0.5) * 0.01
                aug_angle = angle - correction
                if aug_angle < -1.0:
                    aug_angle = -1.0
                angles.append(aug_angle)

    return angles

angles = []

# read udacity's data
angles = read_csv_samples(angles, "data")

# read my recorded data
MYDATA="mydata"
if os.path.isdir(MYDATA) and os.path.exists(MYDATA):
    print("Dir " + MYDATA + " exists")
    for f in os.listdir(MYDATA):
        f_root = os.path.join(MYDATA, f, "data")
        if os.path.isdir(f_root):
            print("read data from", f_root)
            angles = read_csv_samples(angles, f_root, skip=False)

#samples, angles = read_selected_samples(samples, angles)

print("angles ", len(angles))

num_bins = 'auto'
#hist, bin_edges = np.histogram(angles, bins=num_bins)

#print(bin_edges)
#print(hist)

plt.clf()
plt.hist(angles, bins=num_bins)  # plt.hist passes it's arguments to np.histogram
plt.title("Histogram with 'auto' bins")
plt.savefig('1.png')

#samples, angles = read_selected_samples(samples, angles)
#samples, angles = read_selected_samples(samples, angles)

angles = []

# read udacity's data
angles = read_aug_samples(angles, "data")

# read my recorded data
MYDATA="mydata"
if os.path.isdir(MYDATA) and os.path.exists(MYDATA):
    print("Dir " + MYDATA + " exists")
    for f in os.listdir(MYDATA):
        f_root = os.path.join(MYDATA, f, "data")
        if os.path.isdir(f_root):
            print("read data from", f_root)
            angles = read_aug_samples(angles, f_root, skip=False)

print("angles ", len(angles))

plt.clf()
plt.hist(angles, bins=num_bins)  # plt.hist passes it's arguments to np.histogram
plt.title("Histogram with 'auto' bins")
plt.savefig('2.png')

