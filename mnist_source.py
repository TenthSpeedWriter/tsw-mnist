import numpy as np
import tensorflow as tf


class MNIST_Source:
    # The validation set is not used in this example.
    @staticmethod
    def unpickled_data():
        import gzip, cPickle
        source_file = gzip.open('mnist.pkl.gz',
                                'rb')
        source_pickle = cPickle.load(source_file)
        train_set, test_set = (source_pickle[0],
                               source_pickle[2])
        source_file.close()

        return train_set, test_set
    
    #@staticmethod
    #def one_hot_tensor(hot_index, array_size):
    #    hot_list = [0.0 for i in range(array_size)]
    #    hot_list[hot_index] = 1.0
    #    return tf.reshape(hot_list, [1, array_size])
    
    #def tensor_of_one_hots(self, x, num_of_labels):
    #    split_input = tf.split(0, int(np.shape(x)[0]), x)
    #    return tf.concat(0, [self.one_hot_tensor(element.eval(),
    #                                             num_of_labels)
    #                         for element in split_input])
    
    def __init__(self,
                 batch_size=10):
        if 10000 % batch_size != 0:
            err_str = "10000 must be evenly divisible by batch_size (given value {0})".format(str(batch_size))
            raise ValueError(err_str)
        
        IMAGE_SHAPE = (-1, 28, 28, 1)
        self.batch_size = batch_size
        train_set, test_set = self.unpickled_data()
        
        self.train_data = tf.reshape(train_set[0], IMAGE_SHAPE).eval()
        self.train_labels = tf.reshape(test_set[1], [-1, 1])
        
        self.test_data, self.test_labels = (tf.reshape(test_set[0], IMAGE_SHAPE).eval(),
                                            tf.reshape(test_set[1], [-1, 10]).eval())
        
        self._next_train_batch, self._next_test_batch = 0, 0
        self.training_epochs = 0
    
    def next_train_batch(self):
        sliced_img_shape = [28, 28, 1]
        
        data_beginning = [self._next_train_batch, 0, 0, 0]
        data_slice_size = [self.batch_size] + sliced_img_shape
        batch_data = tf.slice(self.train_data, data_beginning, data_slice_size)
        
        label_beginning = [self._next_train_batch, 0]
        label_slice_size = [self.batch_size, 1]
        batch_labels = tf.slice(self.train_labels, label_beginning, label_slice_size)
        
        if self._next_train_batch == 50000 - self.batch_size:
            print "Training epoch reached. Reverting to beginning of training set."
            self._next_train_batch = 0
            self.training_epochs += 1
        else:
            self._next_train_batch += self.batch_size
        return batch_data.eval(), batch_labels.eval()
    
    def next_test_batch(self):
        """
        Returns (test data, test labels) if more data is available, else (-1, -1)
        """
        if self._next_test_batch == 10000:
            print "Test records exhausted."
            return -1, -1
        
        sliced_img_shape = [28, 28, 1]
        
        data_beginning = [self._next_test_batch, 0, 0, 0]
        data_slice_size = [self.batch_size] + sliced_img_shape
        batch_data = tf.slice(self.test_data, data_beginning, data_slice_size)
        
        label_beginning = [self._next_test_batch, 0]
        label_slice_size = [self.batch_size, 1]
        batch_labels = tf.slice(self.test_labels, label_beginning, label_slice_size)
        
        self._next_train_batch += self.batch_size
        return batch_data.eval(), batch_labels.eval()