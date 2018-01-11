**{Machine Learning} **

**Face Gender Recognition **

**With Tensorflow and the Raspberry Pi **

# Info

## Concept

This is project is designated as a learning project in the field of machine learning. The idea here is to use TensorFlow to train a model that can recognize a face as a female or male. This will also be in junction with opencv/tensorflow object detection to find the face in a video feed taken by a raspberry pi 3 camera. Once the gender has been detected, the pi will perform any action the user programs the device to achieve. I this case, the raspberry pi will open the top lid of a toilet seat for a female detected face. In the case of a male detection, the camera will open both lids. Seem interesting enough? Then let's get started!

(for the purpose of this project, the code I will be using is python 3.4 but this can be done in either python 2.7 or 3.4+)

## How it started

I applied for, and was granted access to, a 10k+ dataset of unidentified male and female facial pictures from [www.wilmabrainbridge.com/datasets.html](www.wilmabrainbridge.com/datasets.html).

# STEPS 

**..|..** set up the raspberry pi 3 with raspbian for robots found here: [https://sourceforge.net/projects/dexterindustriesraspbianflavor/](https://sourceforge.net/projects/dexterindustriesraspbianflavor/)

**..|..** Install tensorflow  with virtualenv on your  main computer (I did non gpu installation on a laptop running Ubuntu 17.10). The directions are found here for the virtualenv install: [https://www.tensorflow.org/install/install_linux#InstallingVirtualenv](https://www.tensorflow.org/install/install_linux#InstallingVirtualenv)

**..|..** ssh into your pi  (if its rasbian for robots, it would be ssh pi@dex.local, password: robots1234). then do your sudo apt upgrade and download vnc4server. reset your vnc4server password typing this: vncpasswd 

**..|..** Endure the painstaking time it takes to install OpenCV 3 for python on my raspberry pi. Follow, very closely, the directions found here: [https://www.pyimagesearch.com/2016/04/18/install-guide-raspberry-pi-3-raspbian-jessie-opencv-3/](https://www.pyimagesearch.com/2016/04/18/install-guide-raspberry-pi-3-raspbian-jessie-opencv-3/)

I say painstaking here because all together, this process took about 1.5 hours to complete. the openCv build using all 4 cores took 1 hour by itself, If your pi has trouble overheating, you may have to use less cores which could take even longer. This could also be much longer if you are using a raspberry pi B or B+. GOOD LUCK!

**..|..** Install tensorflow on the raspberry pi  with directions found here: [https://github.com/samjabrahams/tensorflow-on-raspberry-pi/blob/master/GUIDE.md](https://github.com/samjabrahams/tensorflow-on-raspberry-pi/blob/master/GUIDE.md)

**..|..** On the main system, once 2000 male faces had been separated into a folder named male and 2000 faces into a folder named female, I then placed both folders inside of the a folder called faces. (these directions applied to me, but you may be training your system to recognize other objects.) 

**..|..** On the main system, proceed to follow the guide/download Tensorflow for poets from here: [https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0)

**..|..** Follow the same directions for the tensorflow for poets tutorial removing only the folder inside of "tf_files" name "flowers" and replacing it with your folder of pictures. (faces). The time it takes to train your model can vary depending on system and gpu/non gpu. (In my case, I trained the model on and asus ux430 I7 7k with 16 gig memory. It took about 25 minutes to achieve a 94.5 percent accuracy. )

*** helpful hint: If you are looking for more photos to help improve the accuracy of your training model, there is an extension for your chrome browser name FatKun that enables you to do a google image search and easily download all of your results into an appropriately titled folder. It can be found here:  [https://chrome.google.com/webstore/detail/fatkun-batch-download-ima/nnjjahlikiabnchcpehcpkdeckfgnohf?hl=en ](https://chrome.google.com/webstore/detail/fatkun-batch-download-ima/nnjjahlikiabnchcpehcpkdeckfgnohf?hl=en)

**..|..** Once the model has been trained to a percentage that you are happy with, you will then have a python script called label_image.py. This is the python file you run against the face photos when you've detected and have them. You can now test your trained model by running the script with a flag to the location of a photo of a face that hasn't before been seen by your model. 

**..|..** Next we will follow a tutorial to set up and test our picamera. you can find this tutorial here: [https://www.pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python/](https://www.pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python/)

**..|..** After you have tested your camera and know that it is working correctly, it's time to get started using openCV with the picamera. This can sometimes be challenging because there are so many different ways people choose to do this. Since there are so many ways, there are a lot of tutorials which can lead to mis-matching different steps. The tutorial followed here is located at: [https://pythonprogramming.net/raspberry-pi-camera-opencv-face-detection-tutorial/ ](https://pythonprogramming.net/raspberry-pi-camera-opencv-face-detection-tutorial/)

video at: [https://www.youtube.com/watch?v=1I4gHpctXb](https://www.youtube.com/watch?v=1I4gHpctXbU)[U](https://www.youtube.com/watch?v=1I4gHpctXbU)
