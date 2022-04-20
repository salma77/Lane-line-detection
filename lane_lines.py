import numpy as np
import cv2
import matplotlib.pyplot as plt
from binary_thresholding import binary_thresholded
from perspective_transformation import warp

from calculations import *
global left_fit_hist
left_fit_hist = np.array([])

global right_fit_hist
right_fit_hist = np.array([])

def find_lane_pixels_using_histogram(binary_warped, debug=False):
    """
    Finds the lane pixels using a histogram of the image
    Parameters:
        binary_warped: the warped binary image
    Returns:
        leftx, lefty, rightx, righty: the pixel positions of the left and right lane lines
    """
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if debug:
        left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(binary_warped,leftx, lefty, rightx, righty)
        out_img = draw_poly_lines(binary_warped, left_fitx, right_fitx, ploty)
        cv2.imwrite('./debugging/lanes-lines.jpeg', out_img*255)

    return leftx, lefty, rightx, righty


def fit_poly(binary_warped, leftx, lefty, rightx, righty, debug=False):
    """
    Fits a polynomial to the left and right lane lines
    Parameters:
        binary_warped: the warped binary image
        leftx, lefty, rightx, righty: the pixel positions of the left and right lane lines
    Returns:
        left_fit, right_fit: the polynomial coefficients of the left and right lane lines
    """
    ### Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty


    return left_fit, right_fit, left_fitx, right_fitx, ploty


def draw_poly_lines(binary_warped, left_fitx, right_fitx, ploty, debug=False):
    """
    Draws the lane lines on the warped binary image
    Parameters:
        binary_warped: the warped binary image
        left_fitx, right_fitx: the polynomial coefficients of the left and right lane lines
        ploty: the y values of the polynomial fit
    Returns:
        out_img: the image with the lane lines drawn on it
    """
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)

    margin = 100
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array(
        [np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin,
                                                                    ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array(
        [np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin,
                                                                     ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (100, 100, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (100, 100, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # Plot the polynomial lines onto the image
    plt.plot(left_fitx, ploty, color='green')
    plt.plot(right_fitx, ploty, color='blue')
    
    plt.imsave("./debugging/output.jpeg",result, cmap='gray')

    return result


def find_lane_pixels_using_prev_poly(binary_warped, debug=False):
    """
    Finds the pixels of the lane lines using the previous polynomial coefficients
    Parameters:
        binary_warped: the warped binary image
    Returns:
        leftx, lefty, rightx, righty: the pixel positions of the left and right lane lines
    """
    global prev_left_fit
    global prev_right_fit
    # width of the margin around the previous polynomial to search
    margin = 100
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    ### Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    left_lane_inds = ((nonzerox > (prev_left_fit[0]*(nonzeroy**2) + prev_left_fit[1]*nonzeroy +
                                   prev_left_fit[2] - margin)) & (nonzerox < (prev_left_fit[0]*(nonzeroy**2) +
                                                                              prev_left_fit[1]*nonzeroy + prev_left_fit[2] + margin))).nonzero()[0]
    right_lane_inds = ((nonzerox > (prev_right_fit[0]*(nonzeroy**2) + prev_right_fit[1]*nonzeroy +
                                    prev_right_fit[2] - margin)) & (nonzerox < (prev_right_fit[0]*(nonzeroy**2) +
                                                                                prev_right_fit[1]*nonzeroy + prev_right_fit[2] + margin))).nonzero()[0]
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if debug:
        left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(binary_warped,leftx, lefty, rightx, righty)
        out_img = draw_poly_lines(binary_warped, left_fitx, right_fitx, ploty)
        # plt.imshow(out_img)
        # cv2.imwrite('./debugging/lanes-lines.jpeg', out_img*255)
        plt.imsave("./debugging/lane-lines.jpeg",out_img)

    return leftx, lefty, rightx, righty


def project_lane_info(img, binary_warped, ploty, left_fitx, right_fitx, M_inv, left_curverad, right_curverad, veh_pos, debug=False):
    """
    Projects the lane information onto the original image
    Parameters:
        img: the original image
        binary_warped: the warped binary image
        ploty: the y values of the polynomial fit
        left_fitx, right_fitx: the polynomial coefficients of the left and right lane lines
        M_inv: the inverse of the perspective transform matrix
        left_curverad, right_curverad: the radius of curvature of the left and right lane lines
        veh_pos: the vehicle position
    Returns:
        out_img: the image with the lane lines drawn on it
    """
    if debug:
        img = cv2.imread(img)
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(
        color_warp, M_inv, (img.shape[1], img.shape[0]))

    # Combine the result with the original image
    out_img = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    cv2.putText(out_img, 'Curve Radius [m]: '+str((left_curverad+right_curverad)/2)[
                :7], (40, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out_img, 'Center Offset [m]: '+str(veh_pos)[:7], (40, 150),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.6, (255, 255, 255), 2, cv2.LINE_AA)

    return out_img

def lane_finding_pipeline(img, debug=False):
    """
    Pipeline for lane finding
    Parameters:
        img : Input image
    Returns:
        out_img : Output image
    """
    global left_fit_hist
    global right_fit_hist
    global prev_left_fit
    global prev_right_fit
    
    binary_thresh = binary_thresholded(img, debug)
    binary_warped, M_inv = warp(binary_thresh, debug)
    #out_img = np.dstack((binary_thresh, binary_thresh, binary_thresh))*255
    if (len(left_fit_hist) == 0):
        leftx, lefty, rightx, righty = find_lane_pixels_using_histogram(
            binary_warped, debug)
        left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(
            binary_warped, leftx, lefty, rightx, righty, debug)
        # Store fit in history
        left_fit_hist = np.array(left_fit)
        new_left_fit = np.array(left_fit)
        left_fit_hist = np.vstack([left_fit_hist, new_left_fit])
        right_fit_hist = np.array(right_fit)
        new_right_fit = np.array(right_fit)
        right_fit_hist = np.vstack([right_fit_hist, new_right_fit])
    else:
        prev_left_fit = [np.mean(left_fit_hist[:, 0]), np.mean(
            left_fit_hist[:, 1]), np.mean(left_fit_hist[:, 2])]
        prev_right_fit = [np.mean(right_fit_hist[:, 0]), np.mean(
            right_fit_hist[:, 1]), np.mean(right_fit_hist[:, 2])]
        leftx, lefty, rightx, righty = find_lane_pixels_using_prev_poly(
            binary_warped, debug)
        if (len(lefty) == 0 or len(righty) == 0):
            leftx, lefty, rightx, righty = find_lane_pixels_using_histogram(
                binary_warped)
        left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(
            binary_warped, leftx, lefty, rightx, righty)

        # Add new values to history
        new_left_fit = np.array(left_fit)
        left_fit_hist = np.vstack([left_fit_hist, new_left_fit])
        new_right_fit = np.array(right_fit)
        right_fit_hist = np.vstack([right_fit_hist, new_right_fit])

        # Remove old values from history
        if (len(left_fit_hist) > 10):
            left_fit_hist = np.delete(left_fit_hist, 0, 0)
            right_fit_hist = np.delete(right_fit_hist, 0, 0)

    left_curverad, right_curverad = measure_curvature_meters(
        binary_warped, left_fitx, right_fitx, ploty, debug)
    #measure_curvature_meters(binary_warped, left_fitx, right_fitx, ploty)
    veh_pos = measure_position_meters(binary_warped, left_fit, right_fit, debug)
    out_img = project_lane_info(img, binary_warped, ploty, left_fitx,
                                right_fitx, M_inv, left_curverad, right_curverad, veh_pos, debug)
    if debug:
        # cv2.imwrite('./debugging/output.jpeg', out_img)
        plt.imsave("./debugging/output.jpeg",out_img, cmap='gray')
    return out_img

