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

[//]: # (Image References)

[image1]: ./output_images/undist.png "Undistorted"
[image2]: ./output_images/test2.jpg "Road Transformed"
[image3]: ./output_images/combined_binary.jpg "Binary Example"
[image4]: ./output_images/warped.jpg "Warp Example"
[image5]: ./output_images/test3.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `CorrectingForDistortion.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

Using the glob module, I enumerate a list of images of a chessboard taken at different angles, from which I am able to capture object and image points, which are ultimately used to calculate a camera matrix. This matrix is then applied to every input image in order to correct for camera distortion.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image at the start of my Pipeline.execute() function. I use a saturation and direction threshold that are "anded" together. I then "or" this binary with a sobel and hue threshold that have been "anded" together.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I performed a top-view perspective transform after developing a binary image of the combined thresholds of interest. This occurs on line 69 of Pipeline.py.

I take source points that are an estimate of around where the corners of the polygon formed by lane lines would be. The destination points were arbitrarily chosen a certain distance apart, and were strategically chosen so that straight lines would appear parallel after the transform.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

After the perspective transform, I apply the fit_polynomial() function to search for lane lines if lane lines have not already been detected in a previous image. For video streams, I have two instances of the Line class that are tracking certain information from previous frames, which can be used to hone in on where lane lines might be based on previous images. If this data is present, then the fit_polynomial() function is omitted, and the search_around_poly() function is utilized instead. This aspect of the code appears in Pipeline.execute() starting on line 98.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I use measure_curvature_real() within my radius_curve.py file in order to compute a radius of curvature of each line.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step starting on line 117 of Pipeline.execute(). I create a blank image for which I can fill a polygon based on the two polynomials I have calculated to represent each lane. Then I unwarp the perspective to return to the original view of the camera. Then I stack this image of the lane polygon on top of the original image.
![alt text][image5]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./test_videos_output/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

When I have more time to refine my Pipeline, I would like to do more experimentation with color spaces. Specifically, I would like to be able to handle yellow lines better. I also can add more sanity checking, so that it I can avoid recognizing cracks in the middle of the road as lane lines, like in my output for test1.jpg. 
