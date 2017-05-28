import csv
import copy
import os
import cv2
import numpy as np
#import matplotlib.pyplot as plt
from shutil import copyfile

GEN_DIR="gendir_tmp"

def gen_file_name(src, dst, append="", prefix=""):
    head, tail = os.path.split(src)
    if tail == "":
        print("input should be file not path: ", src)
        return ""
    name, ext = os.path.splitext(tail)
    if append != "":
        append = "_" + append
    if prefix != "":
        prefix = prefix + "_"
    return os.path.join(dst, prefix + name + append + ext)

def gen_file(src, dst, angle, flip=False, text=True):
    img = cv2.imread(src)
    if flip == True:
        img = cv2.flip(img, 1)
    if text == True:
        cv2.putText(img, "angle: %.4f" % angle, (30, 30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0))
    cv2.imwrite(dst, img)

def crop_file(src, dst):
    img = cv2.imread(src)
    img = img[56:140,:,:]
    cv2.imwrite(dst, img)

if os.path.exists(GEN_DIR) == False:
    os.mkdir(GEN_DIR) 
if os.path.exists(os.path.join(GEN_DIR, "IMG")) == False:
    os.mkdir(os.path.join(GEN_DIR, "IMG"))

def read_csv_samples(samples, angles, path, skip = True, csv_name='driving_log.csv'):
    count = len(angles)
    print("number of records :", count)

    with open (os.path.join(path, csv_name)) as csvfile:
        reader = csv.reader(csvfile)
        for sample in reader:
            if skip == True:
                skip = False
            else:

                sample[0] = os.path.join(path, sample[0].strip())
                sample[1] = os.path.join(path, sample[1].strip())
                sample[2] = os.path.join(path, sample[2].strip())

                center_name = sample[0].strip()
                left_name = sample[1].strip()
                right_name = sample[2].strip()

                if os.path.exists(sample[0]) == False:
                    print(sample[0], "not exist!")
                    #break
                    continue

                angle = float(sample[3])
                sample[3] = angle

                samples.append(sample)
                angles.append(angle)

                gen_center_name = gen_file_name(center_name, GEN_DIR, prefix="%08d" % count)
                copyfile(center_name, gen_center_name)
                gen_center_name = gen_file_name(center_name, GEN_DIR, append="txt", prefix="%08d" % count)
                gen_file(center_name, gen_center_name, angle)
                crop_name = gen_file_name(gen_center_name, GEN_DIR, append="crop")
                crop_file(gen_center_name, crop_name)

                gen_left_name = gen_file_name(left_name, GEN_DIR, prefix="%08d" % count)
                copyfile(left_name, gen_left_name)
                gen_left_name = gen_file_name(left_name, GEN_DIR, append="txt", prefix="%08d" % count)
                gen_file(left_name, gen_left_name, angle)
                crop_name = gen_file_name(gen_left_name, GEN_DIR, append="crop")
                crop_file(gen_left_name, crop_name)

                gen_right_name = gen_file_name(right_name, GEN_DIR, prefix="%08d" % count)
                copyfile(right_name, gen_right_name)
                gen_right_name = gen_file_name(right_name, GEN_DIR, append="txt", prefix="%08d" % count)
                gen_file(right_name, gen_right_name, angle)
                crop_name = gen_file_name(gen_right_name, GEN_DIR, append="crop")
                crop_file(gen_right_name, crop_name)

                #flip center
                gen_name = gen_file_name(center_name, GEN_DIR, append="flip", prefix="%08d" % count)

                gen_sample = copy.copy(sample)
                gen_sample[0] = gen_name
                gen_sample[3] = -angle
                samples.append(gen_sample)
                angles.append(-angle)

                gen_file(center_name, gen_name, -angle, True)
                crop_name = gen_file_name(gen_name, GEN_DIR, append="crop")
                crop_file(gen_name, crop_name)

                # gen left
                gen_sample = copy.copy(sample)
                gen_sample[0] = sample[1]
                correction = 0.15 + (np.random.random() - 0.5) * 0.01
                gen_sample[3] = sample[3] + correction
                if gen_sample[3] > 1.0:
                    gen_sample[3] = 1.0
                samples.append(gen_sample)
                angles.append(gen_sample[3])

                gen_name = gen_file_name(left_name, GEN_DIR, append="gen", prefix="%08d" % count)
                gen_file(left_name, gen_name, gen_sample[3], False)
                crop_name = gen_file_name(gen_name, GEN_DIR, append="crop")
                crop_file(gen_name, crop_name)

                # gen right
                gen_sample = copy.copy(sample)
                gen_sample[0] = sample[2]
                correction = 0.15 + (np.random.random() - 0.5) * 0.01
                gen_sample[3] = sample[3] - correction
                if gen_sample[3] < -1.0:
                    gen_sample[3] = -1.0
                samples.append(gen_sample)
                angles.append(gen_sample[3])

                gen_name = gen_file_name(right_name, GEN_DIR, append="gen", prefix="%08d" % count)
                gen_file(right_name, gen_name, gen_sample[3], False)
                crop_name = gen_file_name(gen_name, GEN_DIR, append="crop")
                crop_file(gen_name, crop_name)
                count += 1

    return samples, angles

def read_selected_samples(samples, angles):
    with open ('select2.csv') as csvfile:
        reader = csv.reader(csvfile)
        for sample in reader:
            name = sample[0].strip()
            if os.path.exists(name) == False:
                print(name, "not exist!")
                #break
                continue
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

'''
with open ('gen.csv', 'w') as ofile:
    writer = csv.writer(ofile)
    for sample in samples:
        writer.writerow(sample)
'''
