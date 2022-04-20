import cv2
import numpy as np
import matplotlib.pyplot as plt

from camera_calibration import distortion_factors

mtx, dist = distortion_factors()

def warp(img, debug=False):
    """
    Function that applies a perspective transformation to an image
    Parameters:
        img : Image to be transformed
    Returns:
        img_warped : Transformed image
        M_inv : Inverse transformation matrix
    """
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    img_size = (img.shape[1], img.shape[0])
    offset = 300

    # Source points taken from images with straight lane lines, these are to become parallel after the warp transform
    src = np.float32([
        (190, 720),  # bottom-left corner
        (596, 447),  # top-left corner
        (685, 447),  # top-right corner
        (1125, 720)  # bottom-right corner
    ])
    # Destination points are to be parallel, taken into account the image size
    dst = np.float32([
        [offset, img_size[1]],             # bottom-left corner
        [offset, 0],                       # top-left corner
        [img_size[0]-offset, 0],           # top-right corner
        [img_size[0]-offset, img_size[1]]  # bottom-right corner
    ])
    # Calculate the transformation matrix and it's inverse transformation
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(undist, M, img_size)

    if debug:
        plt.imsave("./debugging/warped.jpeg",warped, cmap='gray')
    return warped, M_inv
