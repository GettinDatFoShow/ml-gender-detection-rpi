#OpenCV module
import cv2
#os module for reading training data directories and paths
import matplotlib
#numpy to convert python lists to numpy arrays as it is needed by OpenCV face recognizers
import numpy

from matplotlib import pyplot as plt

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

img = cv2.imread('../trial5.jpg')
# cv2.imshow('img', img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#Draw a rectangle around every found face
for (x,y,w,h) in faces:
    cv2.rectangle(gray,(x,y),(x+w,y+h),(255,255,0),2)
#
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
