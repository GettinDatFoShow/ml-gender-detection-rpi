# This script will detect faces via your webcam.
# Tested with OpenCV3
import io
import time
import picamera
import cv2
import numpy as np
import cv2

# cap = cv2.VideoCapture(0)

# Create the haar cascade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while(True):
	# Create the in-memory stream
	stream = io.BytesIO()
	with picamera.PiCamera() as camera:
		camera.start_preview()
		time.sleep(2)
		camera.capture(stream, format='jpeg')
	# Construct a numpy array from the stream
	data = np.fromstring(stream.getvalue(), dtype=np.uint8)
	# "Decode" the image from the array, preserving colour
	image = cv2.imdecode(data, 1)
	# OpenCV returns an array with data in BGR order. If you want RGB instead
	# use the following...
	frame = image[:, :, ::-1]

	# Capture frame-by-frame
	# ret, frame = cap.read()

	# Our operations on the frame come here
	# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
	# # Detect faces in the image
	# faces = faceCascade.detectMultiScale(
	# 	gray,
	# 	scaleFactor=1.1,
	# 	minNeighbors=5,
	# 	minSize=(30, 30)
	# 	#flags = cv2.CV_HAAR_SCALE_IMAGE
	# )
    #
	# print("Found {0} faces!".format(len(faces)))
    #
	# # Draw a rectangle around the faces
	# for (x, y, w, h) in faces:
	# 	cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


	# Display the resulting frame
	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()