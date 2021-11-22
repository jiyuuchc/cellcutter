import tensorflow as tf

class BatchConv2D(tf.keras.layers.Layer):
    def __init__(self, num_filters, size = 3, activation = 'relu', name=None, **kwargs):
        super(BatchConv2D, self).__init__(name=name)
        self._config_dict = {
          'num_filters': num_filters,
          'size': size,
        }
        conv_kwargs = {}
        conv_kwargs.update(kwargs)
        self._conv = tf.keras.layers.Conv2D(num_filters, size, name= 'conv', **conv_kwargs)
        self._activation = tf.keras.layers.Activation(activation, name = activation)
        self._batchnorm = tf.keras.layers.BatchNormalization(name='norm')

    def get_config(self):
        return self._config_dict

    def call(self, inputs, **kwargs):
        x = self._conv(inputs, **kwargs)
        x = self._batchnorm(x, **kwargs)
        x = self._activation(x, **kwargs)
        return x

class UNetDownSampler(tf.keras.layers.Layer):
    def __init__(self, filters, resnet = False, kernel_regularizer=None, bias_regularizer=None, **kwargs):
        super(UNetDownSampler, self).__init__(**kwargs)
        self._config_dict = {
            'filters': filters,
            'resnet': resnet,
            'kernel_regularizer': kernel_regularizer,
            'bias_regularizer': bias_regularizer,
        }
        conv_kwargs = {
            'padding': 'same',
            'kernel_initializer': 'he_normal',
            'kernel_regularizer': kernel_regularizer,
            'bias_regularizer': bias_regularizer,
        }
        self._maxpool = tf.keras.layers.MaxPool2D(name='maxpool'),
        self._conv1 = BatchConv2D(filters, name='conv_norm_1', **conv_kwargs),
        self._conv2 = BatchConv2D(filters, name='conv_norm_2',**conv_kwargs),

    def get_config(self):
        return self._config_dict

    def call(self, inputs, **kwargs):
        x = self._maxpool(inputs, **kwargs)
        shortcut= x
        x = self._conv1(x, **kwargs)
        x = self._conv2(x, **kwargs)
        if self._config_dict['resnet']:
            x = x + shortcut
        return x

class UNetUpSampler(tf.keras.layers.Layer):
    def __init__(self, filters, resnet=False, kernel_regularizer=None, bias_regularizer=None, **kwargs):
        super(UNetUpSampler, self).__init__(**kwargs)
        self._config_dict = {
            'filters': filters,
            'resnet': resnet,
            'kernel_regularizer': kernel_regularizer,
            'bias_regularizer': bias_regularizer,
        }
        conv_kwargs = {
            'padding': 'same',
            'kernel_initializer': 'he_normal',
            'kernel_regularizer': kernel_regularizer,
            'bias_regularizer': bias_regularizer,
        }
        self._upsample = tf.keras.layers.UpSamling2D(name='upsampling'),
        self._conv1 = BatchConv2D(filters, name='conv_norm_1', **conv_kwargs),
        self._conv2 = BatchConv2D(filters, name='conv_norm_2',**conv_kwargs),

    def get_config(self):
        return self._config_dict

    def call(self, inputs, **kwargs):
        x = self._updample(inputs, **kwargs)
        shortcut= x
        x = self._conv1(x, **kwargs)
        x = self._conv2(x, **kwargs)
        if self._config_dict['resnet']:
            x = x + shortcut
        return x
