import tensorflow as tf

class Conv2dEluLayer:
    """
    Represents a single Exponential Linear Unit layer.
    """
    def __init__(self, x, output_features,
                 conv_filter=None, bias=None,
                 conv_window=(5, 5)):
        """
        Must be initialized with tensor x and the desired number of output features.
        
        May be initialized with existing tf.Variables as its conv_filter and bias,
        though conv_window will be ignored in favor of the existing conv_filter shape.
        """
        self.x, self.out_channels, self.conv_window = (x,
                                                       output_features,
                                                       conv_window)
        
        self.conv_filter = conv_filter if conv_filter else self._new_filter()
        self.bias = bias if bias else self._new_bias()
        
        self.conv2d = tf.nn.conv2d(self.x, self.conv_filter,
                                   strides=[1, 1, 1, 1],
                                   padding='SAME')
        
        self.features = self.conv2d + self.bias
        self.y = tf.nn.elu(self.features)
    
    def _new_filter(self):
        input_channels = int(self.x.get_shape()[-1])
        filter_shape = list(self.conv_window) + [input_channels, self.out_channels]
        return tf.Variable(tf.truncated_normal(filter_shape,
                                               mean=0.0,
                                               stddev=0.1))
    
    def _new_bias(self):
        return tf.Variable(tf.constant(0.1,
                                       shape=[self.out_channels]))