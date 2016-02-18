import tensorflow as tf


class ClassifierNetwork:
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
              learn_rate=tf.constant(1e-2),
              keep_prob=tf.constant(0.5)):
        """
        Runs a single training pass on this network given the specified data and rubric.
        """
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
        """
        feed_dict = {
            self.x: data,
            self.rubric: rubric,
            self.keep_prob: keep_prob
        }
        return self.accuracy.eval(feed_dict=feed_dict)