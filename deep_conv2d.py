import tensorflow as tf
from mnist_source import MNIST_Source
from conv2d_elu_layer import Conv2dEluLayer
from softmax_flat_layer import SoftmaxFlatLayer
from classifier_network import ClassifierNetwork


class DeepConv2dMNIST(ClassifierNetwork):
    """
    A deep network of two conv2d elu and one flat softmax layers with specified feature sizes and variable learn rate.
    """
    def __init__(self,
                 first_layer_features=33,
                 second_layer_features=10,
                 learn_rate = tf.constant(1e-2)):
        """
        Initializes a network of this type with the given layer feature sizes and learn rate
        """
        self.x = tf.placeholder('float', [None, 28, 28, 1])
        self.rubric = tf.placeholder('float', [None, 10])
        self.keep_prob = tf.constant(1.0)
        
        conv2d_0 = Conv2dEluLayer(self.x, first_layer_features,
                                  keep_prob=self.keep_prob)
        conv2d_1 = Conv2dEluLayer(conv2d_0.y, second_layer_features,
                                  keep_prob=self.keep_prob)
        classifier = SoftmaxFlatLayer(conv2d_1.y, 10,
                                      keep_prob=self.keep_prob)
        
        self.y = classifier.y
        
        self.cost = self.cross_entropy(self.y, self.rubric)
        self.learn_rate = learn_rate
        self.trainer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.cost)
        
        self.correct_classifications = tf.equal(tf.argmax(self.y, 1),
                                                tf.argmax(self.rubric, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_classifications, 'float'))
    
    
sess = tf.InteractiveSession()
network = DeepConv2dMNIST()
sess.close()