## Writeup Template
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---


### Overview

In my project, my work can be divided mainly into two parts. Firstly, I use the dataset of cars and no cars to train my SVM model. Secondly, I use the SVM model conbining with silding windows techinique as well as other methods to detecte vihicles in images and videos.


### Histogram of Oriented Gradients (HOG)

#### 1. Extracte HOG features from the training images.

The process of extracting HoG features are contained in the 2-12th cells of VehicleDetectionProjectf.ipynb. I created a function called get_hog_features in the 4th cell of the file in reference of the video in the classes.

Besides HoG features, I also need the color features and the histogram of color, so I create two functions called bin_spatial and color_hist in the 7th and 8th cell.

To accept different parameters to tuning the features, I created the single_img_features fucntion to get the picture of one image, based on that, I created extract_featurs to get the features from the big dataset.

The result of getting features was stored in the folder called 'output_images', the file name is 'HoG_outputs.png'.

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and the resulut are shown:
color_space = 'YCrCb'
orient = 9  
pix_per_cell = 8 
cell_per_block = 2 
hog_channel = 'ALL' 
spatial_size = (16, 16)
hist_bins = 16   
spatial_feat = True 
hist_feat = True 
hog_feat = True 
y_start_stop = [200, 700] 


#### 3. Trained a classifier using HOG features.

After I got the featurs of HoG, I used them to train a linear SVM model to dectect vehicles in images or videos.
The code of traing SVM model was stored in the 12th cell of VehicleDetectionProjectf,ipynb.
Firstly, I created an array of features vector and defined the labels vector based on the feature vector.
Then, I splited up the data into randomized training and test sets to get a better model.
Then, I used normalized the data using the API from third packages.
Finally, I fed the data into the SVM model and got the model I want.

### Sliding Window Search

####  Implemented a sliding window search. 

Because of the size of labeld data and the real image. I used the sliding window to dectect vihicles in big frame.
I had computed the span of the region to be searched and the number of pixeles per step in x-direction and y-direction and the numuber of windows in x-direction and y-direction. These method was inculded in the fucntion called slide_window in the 13th cell. 
Then I create the funtion called search_windows accepting the output of slide_window and export the hod_windows.

The result of getting features was stored in the folder called 'output_images', the file name is 'HeatMap.png'.

Also, there are six images in the folder called 'output_images' show the result of dectecting vehicle in images.


I have bulit a heat-map from the  detections derivateing from the sliding windows in order to combine overlapping detections and remove false positives.
The code of building heat-map was stored in 21-23th cells.
I had tuned parameters of threshold and watch the result on the test_video. After serveral times, I found 4 was the best threshold value to remove false positive and multiple detection.
In fact, I do not use intelligent method to determine the threshold value, I just took serveral experiments, wathched the result and  found the value by hands.



---

### Video Implementation

#### 1. Final video output.  
The result of getting features was stored in the folder called 'output_images', the file name is 'out_project_video.mp4'.


### Discussion
#### 1. Issues I had encounted

When I prepared to extrat features from dataset, I was confused by the choice of features. There are three kind of features, besides in each of them, there are many adjustable parametes. It is realy complicated. Finally, I use all of them! Ha-Ha


Possible Improvment:
Fistly, I think tuning parameters with more time and experimence could imporve the roubustness.
Secondly, considerding the illumination variation and tress shade, I think we can deploy some preprocess algorithm for example top-hat to filter these inrerference.

As shown above, I think light condition could caused failuer of dectection.



