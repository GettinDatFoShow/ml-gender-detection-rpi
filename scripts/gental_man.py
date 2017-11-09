import time
import RPi.GPIO as gpio
import picamera as picam
import label_image
gpio.setmode(gpio.BCM)
gpio.cleanup()
gpio.setwarnings(False)

def operate(image_location) :
    #setup

    label_image.load_labels(image_location)

    #run
    while True :

        if True:
            print("open top lid")
        elif True:
            print("open both lids")
        else:
            print("false alarm")

operate('../tf_files/testing/Google_1_Samuel_Lindquist_5_oval.jpg') #male
operate('../tf_files/testing/Google_1_Samuel-Mansour_8_oval.jpg') #female