# Advanced Lane Finding
### Curved Lane Detection by using computer vision techniques such as perspective transform or image thresholding.

The goals / steps of this algolithms are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## Movie
[![ScreenShot](http://img.youtube.com/vi/f9wI35tasjw/0.jpg)](https://www.youtube.com/watch?v=f9wI35tasjw)

[//]: # (Image References)

[image1]: ./output_images/calibration1.jpg "Undistorted"
[image2]: ./output_images/calibration2.jpg "Road Transformed"
[image3]: ./output_images/bird_view.jpg "Bird View Image"
[image4]: ./output_images/thresholding.jpg "Thresholding"
[image5]: ./output_images/histogram_filtering.jpg "Fit Visual"
[image6]: ./output_images/result.jpg "Output"

### Camera Calibration

#### 1. How to compute the camera matrix and distortion coefficients.

The code for this step is contained in the first code cell of the IPython notebook located in `./Advanced_Lane_Finding.ipynb` (or in lines 12 through 67 of the file called `main.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:
![alt text][image1]

###Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]


#### 2. How to perform a perspective transform.

The code for my perspective transform includes a class called `Perspective_Transform`, which appears in lines 206 in the file `main.py` (in the 6rd code cell of the IPython notebook).  The `Perspective_Transform.transform()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32([[490, 482],[810, 482],
                  [1250, 720],[40, 720]])

dst = np.float32([[0, 0], [1280, 0],
                 [1250, 720],[40, 720]])

```
This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 490, 482      | 0, 0        |
| 810, 482      | 1280, 0      |
| 1250, 720     | 1250, 720      |
| 40, 720      | 40, 720        |

![alt text][image3]

#### 3. How to use color transforms, gradients or other methods to create a thresholded binary image.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines from 69 to 146 in `main.py`).
And you can see example of thresholding 16 and 17rd code cell in `Advanced_Lane_Finding.ipynb`

- The S Channel from the HLS color space, with a min threshold of 150 and a max threshold of 255, did a fairly good job of identifying both the white and yellow lane lines.

- Gradient Thresholding could get the white line. But, have tendency to including some noise. So I apply gaussian blur.

Here's an example of my output for this step.
![alt text][image4]

#### 4. How to identify lane-line pixels and fit their positions with a polynomial.
- Identify peaks in a histogram of the image to determine location of lane lines.  
- mask images by peaks you get
- Fitting a polynomial to each lane using the `cal_poly` method in lines 273 in `main.py`


![alt text][image5]

#### 5. How to calculate the radius of curvature of the lane and the position of the vehicle with respect to center.

- I calculate curvature by using `__cal_curvature` method in lines 312 through 322 in `main.py`.   
Calculate the average of the x intercepts from each of the two polynomials position = *(rightx_int+leftx_int)/2*

- I calculate the position of the vehicle by using `add_place_to_image` method in lines 325 through 335 in `main.py`.  
Calculated the distance from center by taking the absolute value of the vehicle position minus the halfway point along the horizontal axis distance_from_center = *abs(image_width/2 - position)*

**Note**  
The distance from center was converted from pixels to meters by multiplying the number of pixels by 3.7/700.

#### 6. Provide an example image of result

I implemented all pipeline in `main.py`. Main method is `process_image` in line 363 in `Line_detector` class. Here is an example of my result in movie:

![alt text][image6]

---

###Discussion

####1. Discuss any problems / issues I faced in your implementation of this project.

For binary thresholding, I used a combination of color and gradient thresholds. But, a gradient thresholding has high rate of failures when there are objects like line in image. So if you need more robust algorithms, please use only color thresholding. And if there are cars in front of cameras, probably you couldn't recognize lane lines correctly.

And for more improving histogram filtering, by considering that starting point of line would be limited (about 100-300 for left line), the range of searching could be narrow.  
