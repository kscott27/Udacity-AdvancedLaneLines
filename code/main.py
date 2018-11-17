import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import glob
from Line import Line
from Pipeline import Pipeline
from CorrectingForDistortion import parse_calibration_images, cal_undistort, corners_unwarp
from process_image import process_image
from HelperFunctions import weighted_img, draw_lines, extrapolateLine
from sliding_window import fit_polynomial
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

objpoints, imgpoints = parse_calibration_images('../camera_cal/calibration*.jpg', 9, 6)

# Make a list of calibration images
images = glob.glob('../test_images/test*.jpg')

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
	image = cv2.imread(fname)

	undist, mtx, dist = cal_undistort(image, objpoints, imgpoints)
	leftLine = Line()
	rightLine = Line()
	pipeline = Pipeline(leftLine, rightLine)
	result = pipeline.execute(undist)

	plt.imshow(result)

	plt.savefig("../output_images/test" + str(idx+1) + ".jpg")

# white_output = '../test_videos_output/project_video.mp4'
# ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
# ## To do so add .subclip(start_second,end_second) to the end of the line below
# ## Where start_second and end_second are integer values representing the start and end of the subclip
# ## You may also uncomment the following line for a subclip of the first 5 seconds
# ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
# leftLine = Line()
# rightLine = Line()
# pipeline = Pipeline(leftLine, rightLine)
# clip1 = VideoFileClip("../project_video.mp4")
# white_clip = clip1.fl_image(pipeline.execute) #NOTE: this function expects color images!!
# white_clip.write_videofile(white_output, audio=False)