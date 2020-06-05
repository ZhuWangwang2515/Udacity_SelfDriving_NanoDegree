# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

I deployed the same model on the two tracks and get good result in both of them. 
So in the zip file I submitted, I renamed the model name and the video name with a suffix to identify track1 and track2.

### Model Architecture and Training Strategy

#### 1. Training data collection

I collected training data by driving the car in traing model in the simulator.
I driven the car three laps in clockwise and then three laps in anticlockwise focuing on the lane keeping and recover from the side.
There are three pictures in the folder I submitted derivated from each camera mounted on the car.

#### 2. An appropriate model architecture has been employed

I build my model in reference to the LeNet-5 model.
My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 64
The architecture of my model:
    1.Proprecess methods: resized the image from cameras to decrease useless information to get a rubousness model
    2.Convolutional layers with 5x5 filter sizes and 24 filter depth
    3.Poolinng layers using MaxPooling2D method
    3.Convolutional layers with 5x5 filter sizes and 36 filter depth
    4.Poolinng layers using MaxPooling2D method
    5.Convolutional layers with 5x5 filter sizes and 48 filter depth
    6.Poolinng layers using MaxPooling2D method
    7.Convolutional layers with 5x5 filter sizes and 64 filter depth
    8.Poolinng layers using MaxPooling2D method
    9.Convolutional layers with 5x5 filter sizes and 64 filter depth
    10.Poolinng layers using MaxPooling2D method
    11.Flatten the output of the last convolutional layers 
    12.Full connected layers with 120 nodes
    13.Full connceted layers with 84 nodes
    14.Full connceted layers with 1 nodes which is the result of the model

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually, I change the parameters of layers in the model to get
a better bahavior.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 
I used the output of the three cameras to train and drive my car in the simulator.
There are two epoches in the model.
I load all images in memory because I have on good enough server.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 