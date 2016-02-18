import tensorflow as tf

class SoftmaxFlatLayer:
    """
    Represents a flat SoftMax layer
    """
    def __init__(self, x, output_features,
                 weight=None, bias=None,
                 keep_prob=tf.constant(1.0)):
        """
        Must be initialized with tensor x and the desired number of output features or classes.
        
        May be initialized with existing tf.Variables for its weight and bias,
        as well as a specified dropout keep_prob (default 1 if feature is not explicitly used).
        """
        x_shape = [int(dim) for dim in x.get_shape()[1:]]
        x_flat_shape = [-1] + [tf.reduce_prod(x_shape).eval().tolist()]
        
        self.x = x
        self.x_flat = tf.nn.dropout(tf.reshape(self.x, x_flat_shape), keep_prob)
        self._output_features = output_features
        
        self.weight = weight if weight else self._new_weight()
        self.bias = bias if bias else self._new_bias()
        
        self.features = tf.matmul(self.x_flat, self.weight) + self.bias
        self.y = tf.nn.softmax(self.features)
    
    def _new_weight(self):
        weight_shape = (int(self.x_flat.get_shape()[-1]), self._output_features)
        return tf.Variable(tf.zeros(weight_shape))
    
    def _new_bias(self):
        bias_shape = (self._output_features)
        return tf.Variable(tf.zeros(bias_shape))