import csv
import copy
import os
import cv2
import numpy as np
#import matplotlib.pyplot as plt

GEN_DIR="gendir"

if os.path.exists(GEN_DIR) == False:
    os.mkdir(GEN_DIR) 
if os.path.exists(os.path.join(GEN_DIR, "IMG")) == False:
    os.mkdir(os.path.join(GEN_DIR, "IMG"))

def read_csv_samples(samples, angles, path, skip = True, csv_name='driving_log.csv'):
    with open (os.path.join(path, csv_name)) as csvfile:
        reader = csv.reader(csvfile)
        for sample in reader:
            if skip == True:
                skip = False
            else:
                center_name = sample[0].strip()
                sample[0] = os.path.join(path, sample[0].strip())
                sample[1] = os.path.join(path, sample[1].strip())
                sample[2] = os.path.join(path, sample[2].strip())
                if os.path.exists(sample[0]) == False:
                    print(sample[0], "not exist!")
                    #break
                    continue
                sample[3] = float(sample[3])
                samples.append(sample)
                angles.append(sample[3])
                if (sample[3] != 0):
                    #print("sample: ", sample)
                    gen_name = os.path.join(GEN_DIR, center_name)
                    #print("gen file: ", gen_name)
                    if os.path.exists(gen_name) == False:
                        img = cv2.imread(sample[0])
                        img = cv2.flip(img, 1)
                        cv2.imwrite(gen_name, img)
                    gen_sample = copy.copy(sample)
                    gen_sample[0] = gen_name
                    gen_sample[3] = -sample[3]
                    samples.append(gen_sample)
                    angles.append(gen_sample[3])
                    #print("gen sample: ", sample[3], -sample[3])
                    #break
    return samples, angles

def read_selected_samples(samples, angles):
    with open ('select2.csv') as csvfile:
        reader = csv.reader(csvfile)
        for sample in reader:
            name = sample[0].strip()
            if os.path.exists(name) == False:
                print(name, "not exist!")
                break
            sample[3] = float(sample[3])
            samples.append(sample)
            angles.append(sample[3])
            gen_name = os.path.join(GEN_DIR, os.path.basename(name))
            if os.path.exists(gen_name) == False:
                img = cv2.imread(sample[0])
                img = cv2.flip(img, 1)
                cv2.imwrite(gen_name, img)
            gen_sample = copy.copy(sample)
            gen_sample[0] = gen_name
            gen_sample[3] = -sample[3]
            samples.append(gen_sample)
            angles.append(gen_sample[3])
    return samples, angles

samples = []
angles = []

# read udacity's data
#samples, angles = read_csv_samples(samples, angles, "data")

# read my recorded data
MYDATA="mydata"
if os.path.isdir(MYDATA) and os.path.exists(MYDATA):
    print("Dir " + MYDATA + " exists")
    for f in os.listdir(MYDATA):
        f_root = os.path.join(MYDATA, f, "data")
        if os.path.isdir(f_root):
            print("read data from", f_root)
            samples, angles = read_csv_samples(samples, angles, f_root, skip=False)

samples, angles = read_selected_samples(samples, angles)
#samples, angles = read_selected_samples(samples, angles)
#samples, angles = read_selected_samples(samples, angles)

print("samples ", len(samples))
print("angles ", len(angles))

num_bins = 'auto'
hist, bin_edges = np.histogram(angles, bins=num_bins)

print(bin_edges)
print(hist)

'''
plt.clf()
plt.hist(angles, bins=num_bins)  # plt.hist passes it's arguments to np.histogram
plt.title("Histogram with 'auto' bins")
plt.savefig('1.png')
'''

#samples, angles = read_selected_samples(samples, angles)
#samples, angles = read_selected_samples(samples, angles)

'''
angles = []
for sample in augmented_samples:
    angles.append(sample[3])

plt.clf()
hist, bin_edges = np.histogram(angles, bins=num_bins)
plt.hist(angles, bins=num_bins)  # plt.hist passes it's arguments to np.histogram
plt.title("Histogram with 'auto' bins")
plt.savefig('2.png')
'''

with open ('gen.csv', 'w') as ofile:
    writer = csv.writer(ofile)
    for sample in samples:
        writer.writerow(sample)
