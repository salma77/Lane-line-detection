## Lane Line Detection

The purpose of this project is to detect lanes of a highway and to compute the radius of the road's curvature using a video footage of highway driving taken by the front camera. Curved roads are more difficult to navigate than straight ones. The lane lines must be detected, but the images must also be undistorted, in order to compute the curvature appropriately. For camera calibration and perspective transform to get a bird's eye view of the road, image transformation is required.

## Prerequisites and How to run

- You have to have python2+ installed 
- Run `pip install -r requirements.txt` to install all required packages
- Give the run.sh shell execution privileges by running `chmod +x run.sh`

The script run.sh can be run using the command
```
./run.sh input-video-path output-video-path
```
The script can operate in debugging mode using the command
```
./run.sh -d test-image-path
```
## Table of Contents
The following are the project's pipeline and steps:
1. [Camera Calibration](#camera-calibration)
2. [Use a perspective transform to rectify binary image (“birds-eye view”)](#use-a-perspective-transform-to-rectify-binary-image-birds-eye-view)
3. [Processing Binary Thresholded Images](#processing-binary-thresholded-images)
4. [Lane Lines Detection Using Histogram](#lane-lines-detection-using-histogram)
5. [Detection of Lane Lines Based on Previous Cycle](#detection-of-lane-lines-based-on-previous-cycle)
6. [Calculating Vehicle Position and Curve Radius](#calculating-vehicle-position-and-curve-radius)
7. [Project Lane Delimitations Back on Image Plane and Add Text for Lane Info](#project-lane-delimitations-back-on-image-plane-and-add-text-for-lane-info)
8. [Output Video](#output-video)

## Camera Calibration

When seen via camera lenses, optic distortion is a physical phenomena that occurs in picture capture and causes straight lines to appear slightly bent. The front-facing camera on the car is used to record the highway driving video, and the images are distorted. Each camera's distortion coefficients are unique and can be determined using well-known geometrical shapes.

The main purpose of this step is:
- improve the quality of geometrical measurement by undistorting the images coming from the camera
- calculate the spatial resolution of pixels per meter in x & y directions

## Use a perspective transform to rectify binary image (“birds-eye view”)

A bird's eye view is the best perspective for calculating curvature. This means that the road is seen from above rather than via the vehicle's windshield at an angle.

This perspective transform is calculated with a straight lane situation and the assumption that the lane lines are parallel. For the perspective transform, the source and destination points are determined directly from the image.

Given the source and destination points, OpenCV provides perspective transform methods to calculate the transformation matrix for the images. The bird's eye view perspective transform is done with the `warpPerspective` function.

## Processing Binary Thresholded Images

The goal is to process the image so that the lane line pixels are intact and easy to distinguish from the road. Four transformations are used, followed by a combination.

On the gray-scaled image, the first transformation takes the `x sobel`. This represents the x-direction derivative and aids in the detection of vertical lines. Only the values that exceed a certain threshold are saved.

On the grayscaled image, the second transformation picks the white pixels. Values between 200 and 255 were chosen by trial and error on the given images to define white.

## Lane Lines Detection Using Histogram

On binary thresholded images that have already been undistorted and warped, lane line detection is done. The image is first given a histogram. This means that the pixel values in each column are added together to find the most likely x position for the left and right lane lines.  
This is done on multiple steps:
- Split the image height on multiple windows
- For the bottom window, get the peak x value of the histogram and set the window to have it at its center
- calculate the mean of the nonzero indecies and have the next window's center there
- Repeat for all windows

## Lane Line Polynomial
The previous step yeilds `n` points where `n` is the number of windows, those points construct the lane, we can model the lane as:
- position
- slope
- curvature
which means we need a second degree polynomial at least and we get the coeffecients via the `np.fitpoly` method from numpy.

## Calculating Vehicle Position and Curve Radius
Having the coeffecients of the polynomial function from previous step, we can easily calculate the curvature of the lanes with respect to the car's position (y value).  
$ R = \frac{[1 + (y`(x))^2]^\frac{3}{2}}{|y``(x)|} $
## Project Lane Delimitations Back on Image Plane and Add Text for Lane Info
Now we will use the distortion inverse matrix calculated at the beginning $M^{-1}$ to bring everything back to normal, both the image and the lane lines.  
We already have the image so we'll just get back the lane lines from birdeye view to normal view and plot them on top of one another.

## Output Video
We use moviepy to write the output video to disk with the lanes and the lane info added to it.