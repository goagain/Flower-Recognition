### **Notice** <br>
### This file is a copy of [statement of work](business%20understanding/Statement%20of%20Work%20II.docx)  <br>
# Flower Recognition <br>
### Statement of work <br>

Rui Tang <br>
![](business%20understanding/resources/Picture1.jpg)
#### Abstract <br>
This Statement will show the outlines of flower recognition system, which can detect flower type in a picture and label it correctly. <br>

#### Problem Statement <br>
Our team wants to develop an augmented reality glasses. The glasses can automatically identify items in life and mark them. As part of the function, we hope that this automatic identification system can correctly identify the type of flower. <br>

#### Data Requirement <br>
As we mentioned above, we need a bunch of flower pictures with labels. Fortunately, we found the right data on Kaggle. There are 5 types of flowers and our team has initially processed the data. <br>

#### Algorithm <br>
As a visual recognition system, we expect to use Convolutional neural network with TensorFlow. <br>

A convolutional neural network consists of an input and an output layer, as well as multiple hidden layers. The hidden layers of a CNN typically consist of a series of convolutional layers that convolve with a multiplication or other dot product. The activation function is commonly a RELU layer, and is subsequently followed by additional convolutions such as pooling layers, fully connected layers and normalization layers, referred to as hidden layers because their inputs and outputs are masked by the activation function and final convolution. <br>

In this system, inputs are pixels in each picture, and the output will be the probability of each category. Loss evaluation function will be Categorical Crossentropy. <br>

![](business%20understanding/resources/Picture2.jpg)


#### Validation Process and Optimization <br>
To avoid overfitting or underfitting, cross validation will be used. Firstly, we will split data into training set and test set randomly. After each epoch of training, the model will output the accuracy of both training set and test set. <br>

After training, a curve diagram will be showed to indicate whether the model is overfitting or underfitting. <br>

#### Training without data processing <br>
In this project, TensorFlow and Keras were used to train the model. In first version, data were divided into training dataset and testing dataset. Then with batch size 32 and epoch 20, the first model is trained. However, there was a serious overfitting phenomenon.  Based on the following accuracy curve, before the training accuracy curve and validation accuracy curve cross, it is underfitting. After that, the validation accuracy stays still, but the training accuracy goes to nearly perfect. <br>
![](business%20understanding/resources/overfitting.png)

#### Training with Data Transformation <br>
One efficient way to overcome overfitting problem is to find and input more data. However, there is another way to make more dataset. By translation, rotation, scaling, adding random noise, we can produce more data. Using these methods, we can significantly improve the overfitting problem, as shown in the figure below. <br>
![](business%20understanding/resources/accuracy_curve.png)

#### Usage of Model <br>
After training a decent model, we take advantage of OpenCV to process a video. In each frame, OpenCV will extract the image and resize it to fit the model data format and predict. When the model gets the result, we use OpenCV to insert a label and its precision into the frame.
![](business%20understanding/resources/video_snapshot.png)
![](business%20understanding/resources/video_snapshot2.png)
#### Deployment <br>
Complete in the next part.

#### Resources <br>
https://www.kaggle.com/alxmamaev/flowers-recognition <br>

https://en.wikipedia.org/wiki/Convolutional_neural_network <br>
