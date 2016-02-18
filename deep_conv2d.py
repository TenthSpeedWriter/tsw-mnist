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
        self.rubric = tf.placeholder('float', [None, 1])
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
        
        self.correct_classifications = tf.equal(tf.cast(tf.argmax(self.y, 1), 'float'),
                                                self.rubric)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_classifications, 'float'))
    

print "\nInitializing session.\n"
sess = tf.InteractiveSession()

print "\nConstructing network."
network = DeepConv2dMNIST()

print "\nLoading source data."
source = MNIST_Source()

TEST_BATCHES = 2500
LEARN_RATE_MAX = 1e-2
LEARN_RATE_MIN = 1e-4

sess.run(tf.initialize_all_variables())
for i in range(TEST_BATCHES):
    data, labels = source.next_train_batch()
    print data
    print labels
    tweened_learn_rate = LEARN_RATE_MAX - float(i/TEST_BATCHES)*(LEARN_RATE_MAX - LEARN_RATE_MIN)
    print tweened_learn_rate
    if i%25 == 0 and i != 0:
        print "Iteration {0}\n\tBatch Accuracy: {1}%".format(str(i),
                                                             str(100*network.accuracy_on_set(data, labels)))
    network.train(data, labels)

sess.close()