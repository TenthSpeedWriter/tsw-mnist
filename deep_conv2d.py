import tensorflow as tf
from mnist_source import MNIST_Source
from conv2d_elu_layer import Conv2dEluLayer
from softmax_flat_layer import SoftmaxFlatLayer


class DeepConv2dMNIST:
    """
    A deep network of two conv2d elu and one flat softmax layers with specified feature sizes and variable learn rate.
    """
    def __init__(self,
                 first_layer_features=12,
                 second_layer_features=33,
                 learn_rate = tf.constant(1e-2)):
        self.x = tf.placeholder('float', [None, 28, 28, 1])
        self.rubric = tf.placeholder('float', [None, 10])
        
        conv2d_0 = Conv2dEluLayer(self.x, first_layer_features)
        conv2d_1 = Conv2dEluLayer(conv2d_0.y, second_layer_features)
        classifier = SoftmaxFlatLayer(conv2d_1.y, 10)
        
        self.y = classifier.y
        
        self.cost = self.cross_entropy(self.y, self.rubric)
        self.learn_rate = learn_rate
        self.trainer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.cost)
        
        self.correct_classifications = tf.equal(tf.argmax(self.y, 1),
                                                tf.argmax(self.rubric, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_classifications, 'float'))
        
    @staticmethod
    def cross_entropy(y, rubric):
        """
        Returns a scalar representing cross entropy between two tensors.
        Clips the given y values in the effective range of (0, 1) to prevent runaway cost.
        
        I hypothesize that this will make the training process slower yet more stable; we'll find out.
        """
        clipped_y = tf.clip_by_value(y, 1e-9, 1.0 - 1e-9)
        return -tf.reduce_sum(rubric * tf.log(clipped_y))
    
    def train(self, data, rubric,
              learn_rate=tf.constant(1e-2)):
        """
        Runs a single training pass on this network given the specified data and rubric.
        """
        feed_dict = {
            self.x: data,
            self.rubric: rubric,
            self.learn_rate: learn_rate
        }
        self.trainer.run(feed_dict=feed_dict)
    
    def accuracy_on_set(self, data, rubric):
        """
        Returns the evaluation of the accuracy of this network given the specified data and rubric.
        """
        feed_dict = {
            self.x: data,
            self.rubric: rubric
        }
        return self.accuracy.eval(feed_dict=feed_dict)
    
sess = tf.InteractiveSession()
foo = DeepConv2dMNIST()
sess.close()