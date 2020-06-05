## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./AdvancedLaneFinding.ipynb".  

Consider that I need to run the algoriothm many times to tune the parameters, to avoid re-computer the camera calibration matrix, I use to decorator to decorte the function. If there is not the file called calibraion_data.pickle which derivate from the camera calibration, I will calibrate the camera calibration matrix, otherwise, I just load it.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

The result of distortion correction was stored in the folder called 'output_images', the file name is 'distorted_undistorted_img.png'.threshold_img


### Pipeline (single images)
The pipeline will illustrate the architecture of the whole algorithm in this peoject.

#### 1. Compute the camera calibration matrix or load these matrix.
This part is the same with Camera Calibration as shown above.

#### 2. Create a threshold binary image using color space transform, sobel operator and histogram method.

I created a function called threshold_binary in the 4th cell of the "./AdvancedLaneFinding.ipynb".  
I used a combination of color and gradient thresholds as well as the histogram to generate a binary image.

The result of creating binary picture was stored in the folder called 'output_images', the file name is 'threshold_img.png'.

#### 3.Perspective transform for the coming lane lines detection.

I created a function called bird_tranform in the 6th cell of the "./AdvancedLaneFinding.ipynb". 

I used the opecn package to transform perspective, the function I used containing getPerspectiveTransform which aimed to get the tranform matrix, warpPerspective aimed to transform the input image.

The result of perspective transform was stored in the folder called 'output_images', the file name is 'perspective_img.png'.
I chose the hardcode the source and destination points in the following manner:

```python
    src = np.float32([[width,height-10],
                   [0,height-10],
                   [546,460],
                   [732,460]])
    dst = np.float32([[width,height],
                   [0,height],
                   [0,0],
                   [width,0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 1280, 710     | 1280,720    | 
| 0, 710       | 0,720      |
| 546, 460      | 0,0       |
| 732, 460      | 1280,0     |



#### 4. Identified lane-line pixels, fit their positions with a polynomial and calculated the radius of curvature

I create one class called Line to express the property of lane lines and instantiated two objects called line_lt and line_rt to store the lane lines.

There are two methond decorated with property to calibrate the curvature of the lane line in the img and in the real world.

The process of identify lane line pixels contains two methods. If there are avialble data in the last frame, I campute the lane line pixels based on the last frame with more efficient, otherwise I use sliding windows to obtain the lane line pixels.

After I got the area of lane line, I used 'np.ployfit' to obtain the second order polymomial.Then I used 'left_fitx = left_fit_pixel[0]*ploty**2 + left_fit_pixel[1]*ploty + left_fit_pixel[2]' and so on to get the lane line pixels.

The result of lane line pixels detection was stored in the folder called 'output_images', the file name is 'line_fit.png'.

#### 5. The position of the vehicle with respect to center.

I used the middle point of the image to minus the lane center derivate form the position of the right and left lane line to get the possition of the teh vehicle.

#### 6. The result plotted back down onto the road such that the lane area is identified clearly.

The result was stored in the folder called 'output_images', the file name is 'straight_lines1-2.png' and 'test1-6.png'.

---

### Pipeline (video)


The output of the video was sotred in the home folder called 'out_project_10.mp4'

---
