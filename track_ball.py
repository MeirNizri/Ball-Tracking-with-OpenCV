# use code from https://pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/

import argparse
import cv2
import imutils
import pandas as pd

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", 
				default="input.mp4",
				help="path to the video file")
args = vars(ap.parse_args())

# grab a reference to the video file
vs = cv2.VideoCapture(args["video"])

# Get the video frame dimensions and fps
frame_width = int(vs.get(3))
frame_height = int(vs.get(4))
fps = vs.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter to save output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

# define the lower and upper boundaries of the ball in the HSV color space
ball_color_lower = (10,21,248)
ball_color_upper = (85,255,255)
line_thickness = 1
line_color = (0, 0, 0)

# initialize the list of tracked points and velocities
pts = []
velocities_x = []
velocities_y = []

# keep looping on video frames
while True:
	# Read the next frame
	ret, frame = vs.read()

	# If the frame was not read correctly or we have reached the end of the video, break
	if not ret:
		break
	
	# blur the frame to reduce high frequency noise to focus on objects in the frame
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	# convert the frame to the HSV color space
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

	# construct a mask for the ball color, then perform a series of 
	# dilations and erosions to remove any small blobs left in the mask
	mask = cv2.inRange(hsv, ball_color_lower, ball_color_upper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current (x,y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None

	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and centroid
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		# draw circle surrounding the ball
		if radius > 10:
			cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 0), 2)

	# update the list of tracked points and velocities
	if center:
		pts.append(center)
		if len(pts) > 1:
			velocities_x.append((pts[-1][0] - pts[-2][0]) * fps)
			velocities_y.append((pts[-1][1] - pts[-2][1]) * fps)
		else:
			velocities_x.append(0)
			velocities_y.append(0)

    # loop over the set of tracked points
	for i in range(1, len(pts)):
		# if either of the tracked points are None, ignore them
		if pts[i - 1] is None or pts[i] is None:
			continue
		# otherwise, compute the thickness of the line and draw the connecting lines
		cv2.line(frame, pts[i - 1], pts[i], line_color, line_thickness)
	
	# Write the output frame to the output video
	out.write(frame)

	# show the frame to our screen
	cv2.imshow("Frame", frame)
	
	# Check if the user pressed the 'q' key
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# create dataframe from list of tracked points and save it to csv file
pts_df = pd.DataFrame(pts, columns=['Position-x', 'Position-y'])
pts_df['Velocity-x'] = velocities_x
pts_df['Velocity-y'] = velocities_y
pts_df.to_csv('output.csv', index=False)

# release and close all windows
vs.release()
out.release()
cv2.destroyAllWindows()
