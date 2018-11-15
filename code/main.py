import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from Pipeline import pipeline
from CorrectingForDistortion import parse_calibration_images, cal_undistort, corners_unwarp
from process_image import process_image
from HelperFunctions import weighted_img, draw_lines, extrapolateLine
from sliding_window import fit_polynomial
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

objpoints, imgpoints = parse_calibration_images('../camera_cal/calibration*.jpg', 9, 6)

image = cv2.imread("../test_images/test3.jpg")

undist, mtx, dist = cal_undistort(image, objpoints, imgpoints)

result = pipeline(image)

plt.imshow(result)

plt.savefig("../output_images/final_img.jpg")

