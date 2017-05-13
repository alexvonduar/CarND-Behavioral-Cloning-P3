import csv
import copy
import os
import cv2
import numpy as np
import shutil as sh

GEN_DIR="select"

if os.path.exists(GEN_DIR) == False:
    os.mkdir(GEN_DIR)
#if os.path.exists(os.path.join(GEN_DIR, "IMG")) == False
#    os.mkdir(os.path.join(GEN_DIR, "IMG"))

def read_csv_samples(samples, path, skip = True):
    with open (os.path.join(path, 'driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for sample in reader:
            if skip == True:
                skip = False
            else:
                src_name = sample[0].strip()
                src_file = os.path.join(path, src_name)
                if os.path.exists(src_file) == False:
                    print(src_file, "not exist!")
                else:
                    angle = float(sample[3])
                    if angle != 0:
                        if angle > 0:
                            post = "_P"
                        else:
                            post = "_N"
                        dst_name = os.path.basename(src_name)
                        root, ext = os.path.splitext(dst_name)
                        dst_file = os.path.join(GEN_DIR, root + post + ext)
                        sh.copy(src_file, dst_file)
                        sample[0] = dst_file
                        samples.append(sample)
                        #break
    return samples

samples = []

# read udacity's data
samples = read_csv_samples(samples, "data")

# read my recorded data
MYDATA="mydata_dir"
if os.path.isdir(MYDATA) and os.path.exists(MYDATA):
    print("Dir " + MYDATA + " exists")
    for f in os.listdir(MYDATA):
        f_root = os.path.join(MYDATA, f, "data")
        if os.path.isdir(f_root):
            print("read data from", f_root)
            samples = read_csv_samples(samples, f_root, skip=False)

with open ('select.csv', 'w') as ofile:
    writer = csv.writer(ofile)
    for sample in samples:
        writer.writerow(sample)
