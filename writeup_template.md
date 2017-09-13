# **Behavioral Cloning** 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)


[image2]: ./Writeup_material/Model%20Visualization.PNG "Data Visualization"
[image3]: ./Writeup_material/Data%20distrubution.PNG "Data distrubution"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model. This alose exists as a notebook in the file CarND_Behavioral_Cloning_keras.ipynb
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The architecture of the model can be seen in line 232 to 254. It consist of the following layers: 

* Normalization
* Cropping 
* Convolution (24, 5, 5), a stride of (2, 2) and a relu activation function
* Dropout of 0.5
* Convolution (36, 5, 5), a stride of (2, 2) and a relu activation function
* Dropout of 0.5
* Convolution (48, 5, 5), a stride of (2, 2) and a relu activation function
* Convolution (64, 3, 3), a stride of (1, 1) and a relu activation function
* Fully connected Layer (100)
* Fully connected Layer (50)
* Fully connected Layer (10)
* Fully connected Layer (1)


#### 2. Attempts to reduce overfitting in the model

The model contains two dropout layers to reduce overfitting and the training was canceled at 6 epochs, rather then 10.


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 267).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and the track was driven both clock-wise and counter clock-wise. All data wich had a label not equal to zero was flipped and given its negative label. 


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try out larger models and see that they do not overfit. When the problem of the controller was handeled a smaller model was chosen in the end. This was partly to reduce overfitting and to increse the speed of the model. 

The architecture as a whole was always similar. First off I started with a larger patch and larger strides to get some dimentionallity reduction. Then I keept increseing the featuremaps of the model, as this is a good approach to get a good model with fewer parameters. I mainly played around with amount of layers and reduced the amount to get an incresed real-time performance. 

After this I used some fully connected layers, which was reduced by a large amount from the first iterations of the network. 

The data was split into training and validation sets to test the models performance during training. The loss was mainly osberved. 

Many iterations was executd with a look at the simulator for each model. The challenges of slow turning and recovery was tuned with more data. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The architecture of the model can be seen in line 232 to 254. It consist of the following layers: 

* Normalization
* Cropping 
* Convolution (24, 5, 5), a stride of (2, 2) and a relu activation function
* Dropout of 0.5
* Convolution (36, 5, 5), a stride of (2, 2) and a relu activation function
* Dropout of 0.5
* Convolution (48, 5, 5), a stride of (2, 2) and a relu activation function
* Convolution (64, 3, 3), a stride of (1, 1) and a relu activation function
* Fully connected Layer (100)
* Fully connected Layer (50)
* Fully connected Layer (10)
* Fully connected Layer (1)


#### 3. Creation of the Training Set & Training Process

To understand the pipline of the model I started with collecting two laps of training data only using the center lane. As this model performed poorly I collected two more laps clock-wise, one with a lot of recovery and two more laps counter clock-wise. This seemed to be enough data to not overfit the model and generalize to the track. 

Data was augumented by flipping the turning images with its negative label. Below you can see some example images with labels: 

![alt text][image2]

I decided to reduce to number of data-points with a label of 0.0 with 90%. This helped the model a lot wth training and turning in the sumulator. Below you can see the data distrubution: 

![alt text][image3]

