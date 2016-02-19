import numpy as np
import tensorflow as tf

float64 = np.float64

class ClassifierNetwork:
    @staticmethod
    def cross_entropy(y, rubric):
        """
        Returns a scalar representing cross entropy between two tensors.
        Clips the given y values in the effective range of (0, 1) to prevent runaway cost.
        
        I hypothesize that this will make the training process slower yet more stable; we'll find out.
        """
        clipped_y = tf.clip_by_value(y, float64(1e-9), (float64(1.0) - float64(1e-9)))
        return -tf.reduce_sum(rubric * tf.log(clipped_y))
    
    def train(self, data, rubric,
              learn_rate=1e-2,
              keep_prob=1.0):
        """
        Runs a single training pass on this network given the specified data and rubric.
        Allows for a specified keep_prob in (0, 1] to reduce overfitting.
        """
        if keep_prob <= 0 or keep_prob > 1:
            err_str = "Keep prob must be >0 and <= 1 (given {0})".format(str(keep_prob))
            raise ValueError(err_str)
            
        feed_dict = {
            self.x: data,
            self.rubric: rubric,
            self.learn_rate: learn_rate,
            self.keep_prob: keep_prob
        }
        self.trainer.run(feed_dict=feed_dict)
    
    def accuracy_on_set(self, data, rubric):
        """
        Returns the evaluation of the accuracy of this network given the specified data and rubric.
        Fixes keep_prob at 100% for full network analysis.
        """
        feed_dict = {
            self.x: data,
            self.rubric: rubric,
            self.keep_prob: 1.0
        }
        return self.accuracy.eval(feed_dict=feed_dict)