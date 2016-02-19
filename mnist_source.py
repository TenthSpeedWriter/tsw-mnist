import numpy as np


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
        
        self.train_data = np.reshape(train_set[0], IMAGE_SHAPE)
        self.train_labels = np.reshape(test_set[1], [-1, 1])
        
        self.test_data = np.reshape(test_set[0], IMAGE_SHAPE)
        self.test_labels = np.reshape(test_set[1], [-1, 10])
        
        self._next_train_batch, self._next_test_batch = 0, 0
        self.training_epochs = 0
    
    def next_train_batch(self):
        this_start = self._next_train_batch
        next_start = this_start + self.batch_size
        
        batch_data = self.train_data[this_start:next_start]
        batch_labels = self.train_labels[this_start:next_start]
        
        if self._next_train_batch == 50000 - self.batch_size:
            print "Training epoch reached. Reverting to beginning of training set."
            self._next_train_batch = 0
            self.training_epochs += 1
        else:
            self._next_train_batch += self.batch_size
        return batch_data, batch_labels
    
    def next_test_batch(self):
        """
        Returns (test data, test labels) if more data is available, else (-1, -1)
        """
        if self._next_test_batch == 10000:
            print "Test records exhausted."
            return -1, -1
        
        this_start = self._next_train_batch
        next_start = this_start + self.batch_size
        
        batch_data = self.test_data[this_start:next_start]
        batch_labels = self.test_labels[this_start:next_start]
        
        self._next_train_batch += self.batch_size
        
        return batch_data, batch_labels