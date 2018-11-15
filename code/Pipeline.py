import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from radius_curve import measure_curvature_real
from prev_poly import fit_poly, search_around_poly
from sliding_window import find_lane_pixels, fit_polynomial
from GradientHelpers import dir_threshold, mag_thresh

# Edit this function to create your own pipeline.
def pipeline(img, s_thresh=(170, 255), sx_thresh=(15, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_channel = hls[:,:,0]
    h_thresh = (20,35)
    sy_thresh = (0,100)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    mag_binary = mag_thresh(img, sobel_kernel=9, mag_thresh=(10, 255))

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel_x = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    # Sobel y
    sobely = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobely = np.absolute(sobely) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel_y = np.uint8(255*abs_sobely/np.max(abs_sobely))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel_x)
    sxbinary[(scaled_sobel_x >= sx_thresh[0]) & (scaled_sobel_x <= sx_thresh[1])] = 1
    # Threshold y gradient
    sybinary = np.zeros_like(scaled_sobel_y)
    sybinary[(scaled_sobel_y >= sy_thresh[0]) & (scaled_sobel_y <= sy_thresh[1])] = 1
    
    # Threshold s-color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Threshold h-color channel
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    dir_binary = dir_threshold(img, sobel_kernel=9, thresh=(0.7, 1.3))

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(dir_binary)
    # combined_binary[((s_binary == 1) & (h_binary == 1)) | (sxbinary == 1)] = 1
    # combined_binary[(( (sxbinary == 1)) | (dir_binary==1)) | ((s_binary==1) )] = 1
    # combined_binary[(h_binary==1)] = 1
    combined_binary[(((s_binary == 1) & (dir_binary==1)) | ((sxbinary == 1))) | ((h_binary==1))] = 1

    combined_binary = np.uint8(255*combined_binary/np.max(combined_binary))

    plt.imshow(combined_binary)
    plt.savefig("../output_images/combined_binary.jpg")

    # plt.imshow(combined_binary)
    # plt.plot(700, 450, '.')
    # plt.plot(1100, 700, '.')
    # plt.plot(200, 700, '.')
    # plt.plot(600, 450, '.')
    # plt.savefig("../output_images/srcpts.jpg")

    src = np.float32([[700,450],
                      [1100,700],
                      [200,700],
                      [600,450]])

    dst = np.float32([[1000,200],
                      [1000,700],
                      [200,700],
                      [200,200]])

    imshape = combined_binary.shape
    imageHeight = imshape[0]
    imageWidth = imshape[1]
    img_size = (imageWidth, imageHeight)

    # For source points I'm grabbing the outer four detected corners
    # src = np.float32([upperRightVertex, lowerRightVertex, lowerLeftVertex, upperLeftVertex])
    # # For destination points, I'm arbitrarily choosing some points to be
    # # a nice fit for displaying our warped result 
    # # again, not exact, but close enough for our purposes
    # dst = np.float32([[700,400],[700,700],[400,700],[500,400]])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(combined_binary, M, img_size)

    plt.imshow(warped)
    plt.savefig("../output_images/warped.jpg")

    # Run image through the pipeline
    # Note that in your project, you'll also want to feed in the previous fits
    result1, left_fit, right_fit = fit_polynomial(warped)
    result2 = search_around_poly(warped, left_fit, right_fit)

    M_inv = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    unwarped = cv2.warpPerspective(result2, M_inv, img_size)

    return unwarped

