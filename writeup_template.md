# Behavioral Cloning


### Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---
### Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `data.py` containing the script to load data , preprocess image , augment data and the batch generator
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network 
* `writeup_report.md` summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 16 and 64 (model.py lines 23-27) 

The model includes RELU layers to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer (code line 22). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 31,33). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 36).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use an existing, tried and tested model and do the necessary simplications/modifications to suit my requirement. 

I started with the LeNet model and did a few training runs. The simplicity of the model helped me to understand the entire workflow and see how the car was driving with the recorded data. And then I moved to a more powerful NVIDIA model. I finally settled with a simplified version of the NIVIDIA model.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded 2 laps by driving in the opposite direction. This will help in getting data that will balance the steering angles being biased to any one direction. Also it will help to get a new set of data for training.
![alt text][image2]

Due to time constrains, I did not train the model on track two as I was going to test only on the first track and the data from track one provided me good results for track one testing. I will be working on track two after the submission.

To augment the data sat, I also flipped images and angles thinking that this would increase the data available for training. This will also help in balancing the right vs the left steering angles in the sample. For example,here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 6068 number of data points in the driving_log.csv file. I augmented this data using the below techniques.

1. Including Left and Right images : I added the left and right images and adjusted the steering angle by adding +0.25 & -0.25 to the existing steering value.
 I then preprocessed this data by ...
I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

```
Epoch 1/25
75/75 [==============================] - 171s - loss: 0.0169 - val_loss: 0.0145
Epoch 2/25
75/75 [==============================] - 137s - loss: 0.0150 - val_loss: 0.0122
Epoch 3/25
75/75 [==============================] - 101s - loss: 0.0132 - val_loss: 0.0113
Epoch 4/25
75/75 [==============================] - 97s - loss: 0.0125 - val_loss: 0.0110
Epoch 5/25
75/75 [==============================] - 120s - loss: 0.0118 - val_loss: 0.0109
Epoch 6/25
75/75 [==============================] - 134s - loss: 0.0117 - val_loss: 0.0108
Epoch 7/25
75/75 [==============================] - 124s - loss: 0.0114 - val_loss: 0.0108
Epoch 8/25
75/75 [==============================] - 122s - loss: 0.0113 - val_loss: 0.0106
Epoch 9/25
75/75 [==============================] - 157s - loss: 0.0111 - val_loss: 0.0105
Epoch 10/25
75/75 [==============================] - 159s - loss: 0.0110 - val_loss: 0.0104
Epoch 11/25
75/75 [==============================] - 168s - loss: 0.0107 - val_loss: 0.0103
Epoch 12/25
75/75 [==============================] - 106s - loss: 0.0107 - val_loss: 0.0102
Epoch 13/25
75/75 [==============================] - 103s - loss: 0.0103 - val_loss: 0.0101
Epoch 14/25
75/75 [==============================] - 100s - loss: 0.0105 - val_loss: 0.0101
Epoch 15/25
75/75 [==============================] - 107s - loss: 0.0103 - val_loss: 0.0100
Epoch 16/25
75/75 [==============================] - 98s - loss: 0.0101 - val_loss: 0.0099
Epoch 17/25
75/75 [==============================] - 102s - loss: 0.0101 - val_loss: 0.0099
Epoch 18/25
75/75 [==============================] - 113s - loss: 0.0099 - val_loss: 0.0099
Epoch 19/25
75/75 [==============================] - 198s - loss: 0.0098 - val_loss: 0.0098
Epoch 20/25
75/75 [==============================] - 151s - loss: 0.0096 - val_loss: 0.0097
Epoch 21/25
75/75 [==============================] - 119s - loss: 0.0096 - val_loss: 0.0097
Epoch 22/25
75/75 [==============================] - 129s - loss: 0.0094 - val_loss: 0.0096
Epoch 23/25
75/75 [==============================] - 137s - loss: 0.0093 - val_loss: 0.0095
Epoch 24/25
75/75 [==============================] - 136s - loss: 0.0091 - val_loss: 0.0095
Epoch 25/25
75/75 [==============================] - 182s - loss: 0.0090 - val_loss: 0.0095
```