import tensorflow as tf
from mnist_source import MNIST_Source
from conv2d_elu_layer import Conv2dEluLayer
from softmax_flat_layer import SoftmaxFlatLayer

#sess = tf.InteractiveSession()

#x = tf.placeholder('float', [None, 28, 28, 1])

#foo = Conv2dEluLayer(x, 10)
#print foo.y.get_shape()

#bar = SoftmaxFlatLayer(x, 10)
#print bar.y.get_shape()

#sess.close()