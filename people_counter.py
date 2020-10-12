# importing packages
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

# constructing the argument parse and parse the arguments via command line
argp = argparse.ArgumentParser()
argp.add_argument("-p", "--prototxt", required=True,
	help="path of the Caffe 'deploy' prototxt file")
argp.add_argument("-m", "--model", required=True,
	help="path of the Caffe pre-trained model")
argp.add_argument("-i", "--input", type=str,
	help="path to  input video file // optional")
argp.add_argument("-o", "--output", type=str,
	help="path to output video file // optional")
argp.add_argument("-c", "--confidence", type=float, default=0.4,
	help="minimum probability to filter weak detections")
argp.add_argument("-s", "--skip-frames", type=int, default=30,
	help="no of of skip frames between detections")
args = vars(argp.parse_args())

# initialize the list of class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# load our trained model from the system
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# if a video path was not supplied, referencing the webcam
if not args.get("input", False):
	print("[INFO] Starting the video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

# otherwise, grab the video file
else:
	print("[INFO] Opening the specified video file...")
	vs = cv2.VideoCapture(args["input"])

# initialize the video writer for projecting the output onto a file
writer = None

# initialize the frame dimensions (to be set later after reading the first frame of video)
W = None
H = None

# instantiate our optimised centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject to keep the track
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackedObjects = {}

# initialize the total number of frames processed thus far, along with the total number of objects that have moved either up or down with our line
allFrames = 0
totalIn = 0
totalOut = 0

# start the frames per second estimator
fps = FPS().start()

# loop over frames from the video stream
while True:
	# grab the next frame and handle it either video stream or the file specified
	frame = vs.read()
	frame = frame[1] if args.get("input", False) else frame

	# if we are viewing a video and we did not grab a frame then its the end of video
	if args["input"] is not None and frame is None:
		break

	# resize the frame to have a maximum width of 500 pixels (optimising for faster processig), then convert
	# the frame from BGR to RGB for use with dlib
	frame = imutils.resize(frame, width=500)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# if the frame dimensions are empty, initialize them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# if output was specified, initialize
	if args["output"] is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(W, H), True)

	# initialize the current status along with our list of bounding
	# box rectangles returned by either our object detector or correlation trackers
	status = "Waiting"
	rects = []

	# check to see if we are capable of running a more computationally expensive object detection method to assist our tracker
	if allFrames % args["skip_frames"] == 0:
		# set the status and initialize our new set of object trackers
		status = "Detecting"
		trackers = []

		# convert the frame to a blob and pass the blob through the cnn and obtain the detections
		blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
		net.setInput(blob)
		detections = net.forward()

		# loop over the detections to track them
		for i in np.arange(0, detections.shape[2]):
			# extract the confidence (i.e., probability) specified with the prediction obtained
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by requiring a minimum confidence that was passed as argument
			if confidence > args["confidence"]:
				# extract the index of the class label from the detections list
				idx = int(detections[0, 0, i, 1])

				# comparing class label with the prediction gained
				if CLASSES[idx] != "person":
					continue

				# compute the (x, y)-coordinates of the bounding box for the object
				box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
				(startX, startY, endX, endY) = box.astype("int")

				# construct a dlib rectangle object from the bounding
				# box coordinates of the objected detected and then start the dlib correlation
				# tracker to track that object
				tracker = dlib.correlation_tracker()
				rect = dlib.rectangle(startX, startY, endX, endY)
				tracker.start_track(rgb, rect)

				# add the tracker to our list of trackers so we can utilize it during the skipped frames
				trackers.append(tracker)

	# otherwise, we should utilize our object trackers instead of
	# object detectors to obtain a higher frame processing throughput
	else:
		# loop over the trackers
		for tracker in trackers:
			# set the status of our system to be 'tracking' rather

			status = "Tracking"

			# update the tracker and get the updated position
			tracker.update(rgb)
			pos = tracker.get_position()

			# unpack the position object
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			# add the bounding box coordinates to the rectangles list
			rects.append((startX, startY, endX, endY))

	# draw a horizontal line in the center of the frame once an
	# object crosses this line we will determine whether they were
	# going 'outside' or 'inside' the venue
	cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

	# use the centroid tracker to associate the (1) old object
	# centroids with (2) the newly computed object centroids for tracking
	objects = ct.update(rects)

	# loop over the objects tracked 
	for (objectID, centroid) in objects.items():
		# check to see if a trackable object exists for the current
		# object ID
		to = trackedObjects.get(objectID, None)

		# if there is no existing trackable object, create one
		if to is None:
			to = TrackableObject(objectID, centroid)

		# otherwise, there is a trackable object so we can utilize it
		# to get the direction
		else:
			# the difference between the y-coordinate of the current
			# centroid and the mean of previous centroids will tell
			# us in which direction the object is moving (negative for
			# 'outside' and positive for 'inside')
			y = [c[1] for c in to.centroids]
			direction = centroid[1] - np.mean(y)
			to.centroids.append(centroid)

			# check to see if the object was counted or not
			if not to.counted:
				# if the direction is negative (indicating the object
				# is goin outside) AND the centroid is above the center
				# line we count the object
				if direction < 0 and centroid[1] < H // 2:
					totalOut += 1
					to.counted = True

				# if the direction is positive (indicating the object
				# is going inside) AND the centroid is below the
				# center line, count the object
				elif direction > 0 and centroid[1] > H // 2:
					totalIn += 1
					to.counted = True

		# store the trackable object in our dictionary
		trackedObjects[objectID] = to

		# draw both the ID of the object and the centroid of the object on the output frame
		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

	# construct a tuple of information we will be displaying on the output frame
	info = [
		("Going out", totalOut),
		("Going in", totalIn),
		("Status", status),
	]

	# loop over the info tuples and draw them on our frame
	for (i, (k, v)) in enumerate(info):
		text = "{}: {}".format(k, v)
		cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

	# check to see if we should write the output file
	if writer is not None:
		writer.write(frame)

	# Display output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, method to break out of the program
	if key == ord("q"):
		break

	# increment the total number of frames processed thus far and then update the FPS counter
	allFrames += 1
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# check to see if we need to release the video writer pointer to save memory
if writer is not None:
	writer.release()

# if input argument was not given, stop the camera video stream
if not args.get("input", False):
	vs.stop()

# otherwise, release the video file pointer
else:
	vs.release()

# close any open windows
cv2.destroyAllWindows()