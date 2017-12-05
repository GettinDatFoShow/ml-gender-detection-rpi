#OpenCV module
import cv2
#os module for reading training data directories and paths
import matplotlib
#numpy to convert python lists to numpy arrause the face inside ys as it is needed by OpenCV face recognizers
import numpy


from matplotlib import pyplot as plt

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

img = cv2.imread('../trial3.jpg')
# cv2.imshow('img', img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#Draw a rectangle around every found face
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    cv2.imshow('img', img)
    cv2.imshow('face', roi_color)
    cv2.imwrite('../tf_files/testing/found_face.jpg', roi_color)

if(len(faces) < 0 ):
       print("no faces found")
print(faces)
cv2.waitKey(0)
cv2.destroyAllWindows()


