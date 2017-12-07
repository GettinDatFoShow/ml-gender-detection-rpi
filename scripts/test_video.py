import io
import picamera
import cv2
import numpy

# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.vflip = True
camera.resolution = (640, 480)
camera.framerate = 40
rawCapture = PiRGBArray(camera, size=(640, 480))

# allow the camera to warmup
time.sleep(0.1)

# capture frames from the camera
counter = 0
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array

    # show the frame

    # Load a cascade file for detecting faces
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Look for faces in the image using the loaded cascade file
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if len(faces) > 0:
        print("Found " + str(len(faces)) + " face(s) testing in: " + counter)
    else :
        counter = 0
    # Draw a rectangle around every found face

    for (x, y, w, h) in faces:
        counter += 1;
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
        imCrop = image[y:y+h, x:x+w]
        print(y,y+h, x,x+w)
        cv2.imshow("Frame2", imCrop)
        if counter == 5:
            cv2.imwrite("face_output.jpg", imCrop)

    cv2.imshow("Frame", image)


    key = cv2.waitKey(1) & 0xFF

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break