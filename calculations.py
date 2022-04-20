import numpy as np
import cv2


def measure_curvature_meters(binary_warped, left_fitx, right_fitx, ploty, debug=False):
    """
    Calculates the curvature of polynomial functions in meters
    Parameters:
        binary_warped: the warped binary image
        left_fitx, right_fitx: the polynomial coefficients of the left and right lane lines
        ploty: the y values of the polynomial fit
    Returns:
        left_curverad, right_curverad: the curvature of the left and right lane lines in meters
    """
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720  # meters per pixel in y dimension
    xm_per_pix = 3.7/700  # meters per pixel in x dimension

    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix +
                     left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix +
                      right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    if debug:
        print("Left curvature: {} meters".format(left_curverad))
        print("Right curvature: {} meters".format(right_curverad))
        
    return left_curverad, right_curverad


def measure_position_meters(binary_warped, left_fit, right_fit, debug=False):
    """
    Calculates the position of the car in meters
    Parameters:
        binary_warped: the warped binary image
        left_fit, right_fit: the polynomial coefficients of the left and right lane lines
    Returns:
        car_position: the position of the car in meters
    """
    # Define conversion in x from pixels space to meters
    xm_per_pix = 3.7/700  # meters per pixel in x dimension
    # Choose the y value corresponding to the bottom of the image
    y_max = binary_warped.shape[0]
    # Calculate left and right line positions at the bottom of the image
    left_x_pos = left_fit[0]*y_max**2 + left_fit[1]*y_max + left_fit[2]
    right_x_pos = right_fit[0]*y_max**2 + right_fit[1]*y_max + right_fit[2]
    # Calculate the x position of the center of the lane
    center_lanes_x_pos = (left_x_pos + right_x_pos)//2
    # Calculate the deviation between the center of the lane and the center of the picture
    # The car is assumed to be placed in the center of the picture
    # If the deviation is negative, the car is on the felt hand side of the center of the lane
    veh_pos = ((binary_warped.shape[1]//2) - center_lanes_x_pos) * xm_per_pix
    if debug:
        print("Vehicle position: {} meters".format(veh_pos))
    return veh_pos
