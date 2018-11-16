import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from Line import Line
from Pipeline import Pipeline
from CorrectingForDistortion import parse_calibration_images, cal_undistort, corners_unwarp
from process_image import process_image
from HelperFunctions import weighted_img, draw_lines, extrapolateLine
from sliding_window import fit_polynomial
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

# objpoints, imgpoints = parse_calibration_images('../camera_cal/calibration*.jpg', 9, 6)

# image = cv2.imread("../test_images/test3.jpg")

# undist, mtx, dist = cal_undistort(image, objpoints, imgpoints)

# result = pipeline(image)

# plt.imshow(result)

# plt.savefig("../output_images/final_img.jpg")

plt.ion()

white_output = '../test_videos_output/challenge_video.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
leftLine = Line()
rightLine = Line()
pipeline = Pipeline(leftLine, rightLine)
clip1 = VideoFileClip("../challenge_video.mp4")
white_clip = clip1.fl_image(pipeline.execute) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)