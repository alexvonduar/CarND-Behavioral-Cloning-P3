# **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"
[architecture]: ./images/cnn-architecture.png "Model Visualization"
[cropped]: ./images/gen/crop.png "Croped Images"
[gray]: ./images/gen/gray.jpg "Grayscaling"
[img01]: ./images/gen/01.jpg "Left Origin Image"
[img02]: ./images/gen/02.jpg "Left Origin Image cropped"
[img03]: ./images/gen/03.jpg "Left Origin Image with text"
[img04]: ./images/gen/04.jpg "Left Origin Image with text cropped"
[img05]: ./images/gen/05.jpg "Left Origin Image gen"
[img06]: ./images/gen/06.jpg "Left Origin Image gen cropped"
[img11]: ./images/gen/11.jpg "Center Origin Image"
[img12]: ./images/gen/12.jpg "Center Origin Image cropped"
[img13]: ./images/gen/13.jpg "Center Origin Image with text"
[img14]: ./images/gen/14.jpg "Center Origin Image with text cropped"
[img15]: ./images/gen/15.jpg "Center Origin Image flip"
[img16]: ./images/gen/16.jpg "Center Origin Image flip cropped"
[img21]: ./images/gen/21.jpg "Right Origin Image"
[img22]: ./images/gen/22.jpg "Right Origin Image cropped"
[img23]: ./images/gen/23.jpg "Right Origin Image with text"
[img24]: ./images/gen/24.jpg "Right Origin Image with text cropped"
[img25]: ./images/gen/25.jpg "Right Origin Image gen"
[img26]: ./images/gen/26.jpg "Right Origin Image gen cropped"
[augment]: ./images/gen/augment.png "Augmented Images"
[center]: ./images/center.jpg "Drive center"
[left]: ./images/left.jpg "Drive left to center"
[hist_orig]: ./images/orig.png "Original Histogram"
[hist_aug]: ./images/aug.png "Augmented Histogram"

---

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

## **1. Model Architecture**

I've adopted a CNN model based on [NVIDIA's model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/).
The origincal model looks as following
![model architecture][architecture]

and made some changes and preprocessing according to the data input to make it work.

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

My model pipeline have following steps:

### 1. 1 Color Convert and Normalize

Convert input RGB image into grayscale and normalize it to [-0.5, 0.5]. I put this step into file generator:
```python
# function that convert input image into grayscale and then normalize
def normalize_gray(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = np.float64(img)
    return img / 255.0 - 0.5
```

```python
center_image = cv2.imread(name)
center_image = normalize_gray(center_image)
center_image = np.expand_dims(center_image, axis=2)
```
After this step, input 320x160x3 RGB image would be output 320x160x1 Grayscale image.

![gray][gray]

### 1. 2 Crop input image
Crop out the image part that doesn't contain the road, mostly are the sky, far background and front hood. These parts are still(front hood) or changing drastically(sky and background), so crop them out will make our model focusing on the road, that would make it more steady and less computation consumption.

To do this, I choose to skip 56 lines from the top of the image and 20 lines from the bottom of the image. After cropping, 320x160x1 grayscale image would be 320x84x1 grayscale image.

I put this part of code into Keras pipeline as the first layer:

```python
model.add(Cropping2D(cropping=((56,20), (0,0)), input_shape=(160,320,1)))
```

Here's an examplel of cropped image.

![cropped image][cropped]

### 1. 3 Convolution Layers
This part is mostly like NV's original model, only except I add one 3x3 convolutional layer at the end of the convolution part according to the input image size.

```python
model = Sequential()
model.add(Cropping2D(cropping=((56,20), (0,0)), input_shape=(160,320,1)))
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
# conv layer 5
model.add(Convolution2D(128,3,3,activation="relu"))
# conv layer 6
model.add(Convolution2D(160,3,3,activation="relu"))
```

### 1.4 Full Connected Layers

I add a dropout layer with drop rate 0.2 to avoid overfitting with the training data.

```python
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024))
model.add(Dense(512))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
```

### 1.5 Training and Validation Layers

The model used an adam optimizer, so the learning rate was not tuned manually.

```python
model.compile(loss='mse', optimizer='adam')

model.summary()

history = model.fit_generator(train_generator, samples_per_epoch= \
                    len(train_samples), validation_data=validation_generator, \
                                nb_val_samples=len(validation_samples), nb_epoch=1)
```

### 1.6 Whole Pipeline

My final model consisted of the following layers:

|Layer (type)                     |Output Shape          |Param #     |Connected to                     |
|:-------------------------------:|:--------------------:|:----------:|:-------------------------------:|
|cropping2d_1 (Cropping2D)        |(None, 84, 320, 1)    |0           |cropping2d_input_1[0][0]         |
|convolution2d_1 (Convolution2D)  |(None, 80, 316, 24)   |624         |cropping2d_1[0][0]               |
|maxpooling2d_1 (MaxPooling2D)    |(None, 40, 158, 24)   |0           |convolution2d_1[0][0]            |
|convolution2d_2 (Convolution2D)  |(None, 36, 154, 36)   |21636       |maxpooling2d_1[0][0]             |
|maxpooling2d_2 (MaxPooling2D)    |(None, 18, 77, 36)    |0           |convolution2d_2[0][0]            |
|convolution2d_3 (Convolution2D)  |(None, 14, 73, 48)    |43248       |maxpooling2d_2[0][0]             |
|maxpooling2d_3 (MaxPooling2D)    |(None, 7, 37, 48)     |0           |convolution2d_3[0][0]            |
|convolution2d_4 (Convolution2D)  |(None, 5, 35, 64)     |27712       |maxpooling2d_3[0][0]             |
|convolution2d_5 (Convolution2D)  |(None, 3, 33, 128)    |73856       |convolution2d_4[0][0]            |
|convolution2d_6 (Convolution2D)  |(None, 1, 31, 160)    |184480      |convolution2d_5[0][0]            |
|flatten_1 (Flatten)              |(None, 4960)          |0           |convolution2d_6[0][0]            |
|dropout_1 (Dropout)              |(None, 4960)          |0           |flatten_1[0][0]                  |
|dense_1 (Dense)                  |(None, 1024)          |5080064     |dropout_1[0][0]                  |
|dense_2 (Dense)                  |(None, 512)           |524800      |dense_1[0][0]                    |
|dense_3 (Dense)                  |(None, 100)           |51300       |dense_2[0][0]                    |
|dense_4 (Dense)                  |(None, 50)            |5050        |dense_3[0][0]                    |
|dense_5 (Dense)                  |(None, 10)            |510         |dense_4[0][0]                    |
|dense_6 (Dense)                  |(None, 1)             |11          |dense_5[0][0]                    |
|                                 |                      |            |                                 |

## **2. Creation of the Training Set & Training Process**

To capture good driving behavior, I first recorded 4 laps on track one using center lane driving. Here is an example image of center lane driving:

![center lane][center]

And then, I do record and select some short image sequances which drive the car from side to the center. For example:

![left to center][left]

I got total 25135 samples from these collections, and the original histogram of angles are like this:

![original histogram][hist_orig]

To augment the data set, I use two methods: a) flip the image and the angle; b) use left and right camera image with corrections to angle(I choose correction=0.15 together with some random adjustment).

Processing steps are:
* flip image, negate the original angle
* use left camera image, new angle is original angle plus correction with random adjustment
* use right camera image, new angle is original angle minus correction with random adjustment

![Augmented Images][augment]

After that, I had 100540 number of data points, and new histogram is like this:

![augmented histogram][hist_aug]

I finally randomly shuffled the data set and put 20% of the data into a validation set.
```python
np.random.shuffle(samples)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)
```

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 14 since after then the validation error got to oscillate.

```sh
Epoch 1/20
94777/94777 [==============================] - 291s - loss: 0.0077 - val_loss: 0.0057
Epoch 2/20
94777/94777 [==============================] - 287s - loss: 0.0058 - val_loss: 0.0048
Epoch 3/20
94777/94777 [==============================] - 287s - loss: 0.0044 - val_loss: 0.0032
Epoch 4/20
94777/94777 [==============================] - 287s - loss: 0.0036 - val_loss: 0.0030
Epoch 5/20
94777/94777 [==============================] - 286s - loss: 0.0034 - val_loss: 0.0032
Epoch 6/20
94777/94777 [==============================] - 286s - loss: 0.0033 - val_loss: 0.0028
Epoch 7/20
94777/94777 [==============================] - 285s - loss: 0.0031 - val_loss: 0.0028
Epoch 8/20
94777/94777 [==============================] - 286s - loss: 0.0030 - val_loss: 0.0028
Epoch 9/20
94777/94777 [==============================] - 285s - loss: 0.0029 - val_loss: 0.0033
Epoch 10/20
94777/94777 [==============================] - 285s - loss: 0.0029 - val_loss: 0.0027
Epoch 11/20
94777/94777 [==============================] - 285s - loss: 0.0028 - val_loss: 0.0026
Epoch 12/20
94777/94777 [==============================] - 285s - loss: 0.0027 - val_loss: 0.0026
Epoch 13/20
94777/94777 [==============================] - 285s - loss: 0.0029 - val_loss: 0.0026
Epoch 14/20
94777/94777 [==============================] - 285s - loss: 0.0026 - val_loss: 0.0025
Epoch 15/20
94777/94777 [==============================] - 285s - loss: 0.0026 - val_loss: 0.0028
Epoch 16/20
94777/94777 [==============================] - 285s - loss: 0.0026 - val_loss: 0.0026
Epoch 17/20
94777/94777 [==============================] - 284s - loss: 0.0025 - val_loss: 0.0025
Epoch 18/20
94777/94777 [==============================] - 285s - loss: 0.0025 - val_loss: 0.0028
Epoch 19/20
94777/94777 [==============================] - 285s - loss: 0.0024 - val_loss: 0.0026
Epoch 20/20
94777/94777 [==============================] - 285s - loss: 0.0023 - val_loss: 0.0024
```