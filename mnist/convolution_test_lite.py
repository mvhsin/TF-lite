""" Convolutional Neural Network.

Build and train a convolutional neural network with TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)

test_file = mnist.test.images[0]
# Show image that we want to predict
plt.imshow(test_file.reshape((28, 28)))
plt.show()

# Running a test dataset by loading the model saved earlier
with tf.Session() as sess:

    # Load TFLite model and allocate tensors.
        #interpreter = tf.contrib.lite.Interpreter(model_path='tf_lite/model_lite/conv_net_f32.tflite')
        interpreter = tf.contrib.lite.Interpreter(model_path='tf_lite/model_lite/conv_net_uint8.tflite')
        interpreter.allocate_tensors()

    # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

    # Test model with input data
        input_data = np.zeros((1, num_input))
        input_data[0,:] = test_file
        #input_data = input_data.astype(np.float32)
        input_data = input_data.astype(np.uint8)
        print(np.amax(input_data[0,:]))

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]['index'])
        ans = np.argmax(np.asarray(out[0]))
        
        print("Results:", out[0])
        print("Ans:", ans)
