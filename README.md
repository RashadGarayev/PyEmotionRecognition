
# Emotion-recognition

The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. The task is to categorize each face based on the emotion shown in the facial expression in to one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).

train.csv contains two columns, "emotion" and "pixels". The "emotion" column contains a numeric code ranging from 0 to 6, inclusive, for the emotion that is present in the image. The "pixels" column contains a string surrounded in quotes for each image. The contents of this string a space-separated pixel values in row major order. test.csv contains only the "pixels" column and your task is to predict the emotion column.

The training set consists of 28,709 examples. The public test set used for the leaderboard consists of 3,589 examples. The final test set, which was used to determine the winner of the competition, consists of another 3,589 examples.

This dataset was prepared by Pierre-Luc Carrier and Aaron Courville, as part of an ongoing research project. They have graciously provided the workshop organizers with a preliminary version of their dataset to use for this contest.

https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
<a id="p2"></a> 
# Installations:

**Note: Python 2.x is not supported**
<img src="https://camo.githubusercontent.com/ba2171fe9ab58bba2f169b740c35c26bd3cb4241/68747470733a2f2f696d672e736869656c64732e696f2f707970692f707976657273696f6e732f70796261646765732e737667" alt="versions" data-canonical-src="https://img.shields.io/pypi/pyversions/pybadges.svg" style="max-width:100%;">


-keras

-tensorflow-gpu

-opencv-contrib

-numpy

-PyQt5

<a id="p3"></a> 
# Usage:

The program will creat a gui to display the scene capture by webcamera and a window representing the probabilities of detected emotions.

> Run

python emotion.py


<a id="p4"></a> 
# Dataset:

Dataset [install from kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) 


# Training and Test #

Read dataset (csv file)
<p align="center"> <img src="image/pic1.jpg" alt="drawing" width="500"/> </p>


----------
28709 training image in Dataset

<p align="center"> <img src="image/pic2.jpg" alt="drawing" width="500"/> </p>

> <p align="left"> <img src="image/pic3.jpg" alt="drawing" width="500"/> 

> </p> <p align="right"> <img src="image/pic4.jpg" alt="drawing" width="500"/> </p>


----------

3589 test image in Dataset
<p align="left"> <img src="image/pic5.jpg" alt="drawing" width="500"/> </p>

> <p align="left"> <img src="image/pic6.jpg" alt="drawing" width="500"/> </p>

## create Convolutional Neural Network Keras (CNN) ##

> <p align="left"> <img src="image/pic7.jpg" alt="drawing" width="500"/> </p>

> <p align="left"> <img src="image/pic8.jpg" alt="drawing" width="300"/> </p>

----------

- > <p align="left"> <img src="image/pic10.jpg" alt="drawing" width="400"/> </p>


Loss graphics
<p align="left"> <img src="image/loss.jpg" alt="drawing" width="500"/> </p>


- If you have an Nvidia GPU, then you can install `tensorflow-gpu` package. It will make things run a lot faster.
Depending on the hardware configuration of your system, the execution time will vary. On CPU, training will be slow. Please use onlu tensorflow-gpu(Cuda Toolkit and CudNN).


## Help
If any issues and suggestions to me, you can create an  [issue](https://github.com/RashadGarayev/PyEmotionRecognition/issues) or reach out on Facebook [Rashad Garayev](https://www.facebook.com/fly.trion) .



