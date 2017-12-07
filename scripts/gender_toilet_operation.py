from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import picamera
import cv2
import numpy as np

import argparse
import sys

# import the necessary packages
# from picamera.array import PiRGBArray
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import tensorflow as tf
import RPi.GPIO as GPIO
from time import sleep



def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
                                input_mean=0, input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels=3,
                                           name='png_reader')
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                      name='gif_reader'))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
    else:
        image_reader = tf.image.decode_jpeg(file_reader, channels=3,
                                            name='jpeg_reader')
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0);
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label



# def SetTopAngle(angle):
#     duty = angle /18+2
#     GPIO.output(7, True)
#     top.ChangeDutyCycle(duty)
#     sleep(1)
#     GPIO.output(7, False)
#     top.ChangeDutyCycle(0)
#
# def SetBottomAngle(angle):
#     duty = angle /18+2
#     GPIO.output(11, True)
#     bottom.ChangeDutyCycle(duty)
#     sleep(1)
#     GPIO.output(11, False)
#     bottom.ChangeDutyCycle(0)

if __name__ == "__main__":
    file_name = "./face_output.jpg"
    model_file = "../tf_files/retrained_graph.pb"
    label_file = "../tf_files/retrained_labels.txt"
    input_height = 224
    input_width = 224
    input_mean = 128
    input_std = 128
    input_layer = "input"
    output_layer = "final_result"

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="image to be processed")
    parser.add_argument("--graph", help="graph/model to be executed")
    parser.add_argument("--labels", help="name of file containing labels")
    parser.add_argument("--input_height", type=int, help="input height")
    parser.add_argument("--input_width", type=int, help="input width")
    parser.add_argument("--input_mean", type=int, help="input mean")
    parser.add_argument("--input_std", type=int, help="input std")
    parser.add_argument("--input_layer", help="name of input layer")
    parser.add_argument("--output_layer", help="name of output layer")
    args = parser.parse_args()

    if args.graph:
        model_file = args.graph
    if args.image:
        file_name = args.image
    if args.labels:
        label_file = args.labels
    if args.input_height:
        input_height = args.input_height
    if args.input_width:
        input_width = args.input_width
    if args.input_mean:
        input_mean = args.input_mean
    if args.input_std:
        input_std = args.input_std
    if args.input_layer:
        input_layer = args.input_layer
    if args.output_layer:
        output_layer = args.output_layer
    # GPIO.cleanup()


    #
    # GPIO.setmode(GPIO.BOARD)
    # GPIO.setup(7, GPIO.OUT)
    # GPIO.setup(11, GPIO.OUT)
    #
    # top = GPIO.PWM(7, 50)
    # top.start(0)
    # bottom = GPIO.PWM(11, 50)
    # bottom.start(0)


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

        # print("Found " + str(len(faces)) + " face(s)")

        # Draw a rectangle around every found face

        for (x, y, w, h) in faces:
            counter += 1
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
            imCrop = image[y:y + h, x:x + w]
            # print(y, y + h, x, x + w)
            # cv2.imshow("Frame2", imCrop)
            if counter == 10:
                cv2.imwrite("face_output.jpg", imCrop)
                counter = 0

                graph = load_graph(model_file)
                t = read_tensor_from_image_file(file_name,
                                                input_height=input_height,
                                                input_width=input_width,
                                                input_mean=input_mean,
                                                input_std=input_std)

                input_name = "import/" + input_layer
                output_name = "import/" + output_layer
                input_operation = graph.get_operation_by_name(input_name);
                output_operation = graph.get_operation_by_name(output_name);

                with tf.Session(graph=graph) as sess:
                    results = sess.run(output_operation.outputs[0],
                                       {input_operation.outputs[0]: t})
                results = np.squeeze(results)

                top_k = results.argsort()[-5:][::-1]
                labels = load_labels(label_file)
                for i in top_k:
                    print(labels[i], results[i])

        cv2.imshow("Frame", image)

        key = cv2.waitKey(1) & 0xFF

        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            # top.stop()
            # bottom.stop()
            # GPIO.cleanup()
            break