# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Descripmention of my own pipeline.

Firstly, to do more efficient work, I use one 'for' loop to handle all the six picture once time.

My pipeline consisted of 6 steps:

    (1)I inputed the origin pictures, and then converted the images to grayscale and shoow them 
    
    (2)Did gaussian smooth and the kernal size is 5.
    
    (3)Did edge detection by Canny operator.
    
    (4)Defined the interest region on these pictures
    
    (5)Did Hough Transform.
    
    (6)Rename the outputs by automation.
    
![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


When I debuged the challenge work, I found there are too much nosie included teh pavemnt, the light and trees, so I think we need to make preprocessing procedure before the detection, for intance, we can do top-hat transform and histogram equalization.


### 3. Suggest possible improvements to your pipeline

The biggest improment for me is that I need precprocess this videos(pictures in fact) firstly!
