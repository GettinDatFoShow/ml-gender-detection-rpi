This is project is desinated as a learning project in the field of machine learning. The idea here is to use TensorFlow to train a model that can recognize a face as a female or male. This will also be in junction with opencv/tensorflow object detection to find the face in a video feed taken by a raspberry pi 3 camera. Once the gender has been detected, the pi will perform any action the user programs the device to achieve. I this case, the raspberry pi will open the top lid of a toilet seat for a female detected face. In the case of a male detection, the camera will open both lids. Seem interesting enough? Then lets get started!

(for the purpose of this project, the code I will be using is python 3.4 but this can be done in either python2.7 or 3.4+)

First: I applied for, and was granted access to, a 10k+ dataset of unidentified male and female facial pictures frpm www.wilmabrainbridge.com/datasets.html.

Second: set up the raspberry pi 3 with raspbian for robots found here: https://sourceforge.net/projects/dexterindustriesraspbianflavor/

... : Install tensorflow non gpu with virtualenv on my main laptop running Ubuntu 17.10 following the directions found here for the virtualenv install: https://www.tensorflow.org/install/install_linux#InstallingVirtualenv

... : ssh into your pi (if its rasbian for robots, it would be ssh pi@dex.local, password: robots1234.) then do your sudo apt upgrade and download vnc4server. reset your vnc4server password typing this: vncpasswd 

... : install a vnc viewer that you prefer on your main computer. In this case I used vinagre. pretty simple to use, start the vnc4server on your pi by typing that. then it will tell you 'desktop is blah:#' mine was dex:2. when you start vnc on your main computer, it will ask you the host. In this case I typed dex.local:2. That will let you view your pi desktop if your not akin to doing everything on the command line through ssh. 

... : I endured the painstaking time it took to install OpenCV 3 for python on my raspberry pi. I followed, near to exactly, the directions found here: https://www.pyimagesearch.com/2016/04/18/install-guide-raspberry-pi-3-raspbian-jessie-opencv-3/
I say painstaking here becuase all together, this process took about 1.5 hours to complete. the openCv build using all 4 cores took 1 hour by itself. If your pi has trouble overheating, you may have to use less cores which could take even longer. GOOD LUCK! 

... : Installed tensorflow on my raspberry pi 3 with directions found here: https://github.com/samjabrahams/tensorflow-on-raspberry-pi/blob/master/GUIDE.md

... : On the main system, once 2000 male faces had been seperated into a folder named male and 2000 faces into a folder named female, I then placed both folders inside of the a folder caled faces. 

... : On the main system, proceed to follow the guide/download Tensorflow for poets from here: https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0

... : On the main system, removed the flower folders and place in the folder of the models you wish to train. In this case, it is faces.

... : Follow the same directions for the tensflow for poets tutorial removing only the folder inside of "tf_files" name "flowers" and replacing it with your folder of pictures. (faces). The time it takes to train your model can vary depending on system and gpu/non gpu. In my case, I trained the model on and asus ux430 I7 7k with 16 gig memory. It took about 25 minutes to achieve a 94.5 percent accuracy. 

... : Once the model has been trained to a percentage that you are happy with, you will then have a python script called label_image.py. This is the python file you run against the face photos when you've detected and have them. You can now test your trained model by running the script with a flag to the location of a photo of a face that hasnt before been seen by your model. 

... : Next we will follow a tutorial to set up and test our picamera. you can find this tutorial here: https://www.pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python/







