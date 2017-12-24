# Guokai-Advanced-Lane-Lines
## Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

histogram_of_ lower_half_image.png

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[image7]: ./output_images/regionofinterest.png "Region of interest image"
[image8]: ./output_images/undistort_threshold_region_persTran_image.png "Processed image"
[image9]: ./output_images/histogram_of_lower_half_image.png "histogram of image"
[image10]: ./output_images/perspective-transformed_image.png "perspective transform of image"
[image11]: ./output_images/curvature.png "curvature of detected lane lines"
[video1]: ./test_video/output_project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained between lines #7 ~ #44 of the file called `Advanced_Lane_Line.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines #46 through #80 in `Advanced_Lane_Line.py`).  Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines #118 through #142 in the file `Advanced_Lane_Line.py`.  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

As we have developed the first project _[**Lane_Lines-P1**](https://github.com/wizardkeven/CarND-LaneLines-P1)_, I think it may be beneficial to reuse some of the code in this project. I chose 'region of interest' to apply a selected trapezoid mask for lane detection instead of the whole image, which appears in lines #82 through #115 in the file `Advanced_Lane_Line.py`. The trapezoid matrix is as following:

```
# Get input image shape
imshape = image.shape
# vertices for interest region mask
vertices = np.array([[(imshape[1]*0.1,imshape[0]-1),
                      (imshape[1]*0.45, imshape[0]*0.6), 
                      (imshape[1]*0.55, imshape[0]*0.6),
                      (imshape[1]*0.95,imshape[0]-1)]],
                    dtype=np.int32)
```

The mask region is as below:

![alt text][image7]

I verified that my _region of interest_ and _perspective transform_ was working as expected by first mask the _trapezoid_ on the undistorted image and then drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image8]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I looked up to course code and use histogram calculation to determine lane lines positions on bottom of image like this:

![alt text][image10] ![Histogram of lower part image][image9]

and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines #553 through #540 in my code in `Advanced_Lane_Line.py`.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented all those steps above in lines #533 through #570 in my code in `Advanced_Lane_Line.py` in the function `pipe_line()`.  Here is an example of my result on a test image:

![alt text][image11]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./test_video/output_project_video.mp4)
There contains also the output for [challenge_video.mp4](https://github.com/wizardkeven/Guokai-Advanced-Lane-Lines/blob/master/test_video/output_challenge_video.mp4) and [harder_challenge_video.mp4](https://github.com/wizardkeven/Guokai-Advanced-Lane-Lines/blob/master/test_video/output_harder_challenge_video.mp4).

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

I use parameters provided in course code for image undistortion, color and sobel thresholding and also perspective transform and it works well on tested images and captured images. There are not too much to go into detail on this part.

The real challenge is actually the stratagy of how to detect and draw lane lines for video. Because we can't tell how our code performs until applying it on a consecutive images steam with varying lane line curvature, road situation and light condition. I spend quite some time on tuning parameters to fit on test images, but it turns out it not so significant for the real challenge. The first generated video was catastrophic on two turnings with strong light and completely lost tracking of right lane line. This reminds me to consider deeply the lane detection stratagy rather than a half understood stratagy.

At first, I captured the images of video clip where my code detected realy bad(_video clip of 21 sec~22sec_ in project.video). Then I found the output of _right lane line base position_ is completed deviated from base point. So it means I need apply the previous "best fit" to replace the current detected one. Then **how to decide "best fit ?" "how to determine a bad detection? etc"** all these problem arised when it comes to "smoothing lane lines".

In fact, I didn't realize the importance of the given _**line**_ class even how to use it to smoothe the lines at first. In fact, caching previous parameters can help decide the problem as:
* if the current detection fails
* if I need to calculate lane lines from histogram instead of using averaged fit
* if I need dump the tracking parameters and restart
In fact, in answering all these questions, the final stratagy has come out most of it. Here is the protocol I use to make lane lines smoothe:
* Start from histogram detection and initialize all parameters with this result
* if lane line position _**left_base_P**_ or _**right_base_P**_ deviation from past "bestx" is larger than threshold value _**max_dev**_, then the detection for left(or right) fails, use cached _**best_fit**_, and don't update parameters
* if not, then the detection succeeds, use current detected _**fit**_, add current parameters into line, recalculate _**bestx**_ and _**best_fit**_.
* if bad detection rate exceeds 30% but smaller than 80% then calculate lane lines from histogram for next frame.
* if exeeds 80%, dump line parameters
* if caching list for past "Xs" and "fit" exceeds max size _**keep_max**_, remove least recent records.

The threshold values for above pipeline are approached by real experimentations. Max size _**keep_max**_ for caching list and max acceptable deviation _**max_dev**_ are two tricky parameters. If we keep tracking a too long list, the previous values will have too much weight on current detection. In this situation, it will deviate too much if current detection is a perfect one. In other hand, if we give  _**max_dev**_ a low threshold, most of detections would be "fails" and the tracking list will not contain enough correct data to get a "good" averaged value when detection fails.

More details are all included in `Advanced_Lane_Line.py` and it is well commented for read. You can check it to get more information.









