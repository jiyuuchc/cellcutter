import tensorflow as tf
import tensorflow.keras.layers as layers

class BatchConv2D(tf.keras.layers.Layer):
    def __init__(self, num_filters, size = 3, activation = 'relu', name=None, **kwargs):
        super(BatchConv2D, self).__init__(name=name)
        self._config_dict = {
          'num_filters': num_filters,
          'size': size,
          'activation': activation,
          'name': name,
        }
        self._config_dict.update(**kwargs)
        conv_kwargs = {
          'padding': 'same',
        }
        conv_kwargs.update(kwargs)
        self._conv = layers.Conv2D(num_filters, size, name= 'conv', **conv_kwargs)
        self._activation = layers.Activation(activation, name = activation)
        self._batchnorm = layers.BatchNormalization(name='norm')

    def get_config(self):
        return self._config_dict

    def call(self, inputs, **kwargs):
        x = self._conv(inputs, **kwargs)
        x = self._activation(x, **kwargs)
        x = self._batchnorm(x, **kwargs)
        return x
