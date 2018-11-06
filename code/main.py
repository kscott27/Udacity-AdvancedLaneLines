import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from process_image import process_image
from HelperFunctions import weighted_img, draw_lines, extrapolateLine
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

image = mpimg.imread("CarND-Advanced-Lane-Lines/test_images/straight_lines1.jpg")
# final_image = process_image(image)

# final_image, matrix = corners_unwarp(image)
plt.imshow(image)
plt.savefig("CarND-Advanced-Lane-Lines/output_images/original_image.jpg")

objpoints = []
imgpoints = []

objp = np.zeros((9*6, 3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
if ret == True:
	imgpoints.append(corners)
	objpoints.append(objp)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

warped, M = corners_unwarp(image, 9, 6, mtx)
plt.imshow(warped)
plt.savefig("CarND-Advanced-Lane-Lines/output_images/warped_image.jpg")