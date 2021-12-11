import tensorflow as tf
from .common import *

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
        if resnet:
            self._down_conv = tf.keras.layers.Conv2D(filters, 3, strides=2, name='down_conv', **conv_kwargs)
        else:
            self._maxpool = tf.keras.layers.MaxPool2D(name='maxpool')
        self._conv1 = tf.keras.layers.Conv2D(filters, 3, name='conv_1', activation='relu', **conv_kwargs)
        self._conv2 = tf.keras.layers.Conv2D(filters, 3, name='conv_2', activation='relu', **conv_kwargs)
        self._norm  = tf.keras.layers.BatchNormalization(name='norm')

    def get_config(self):
        config = super(UNetDownSampler,self).get_config()
        config.update(self._config_dict)
        return config

    def call(self, inputs, **kwargs):
        if self._config_dict['resnet']:
            x = self._down_conv(inputs, **kwargs)
            shortcut = x
            x = tf.keras.activations.relu(x)
        else:
            x = self._maxpool(inputs, **kwargs)
            shortcut = x
        x = self._conv1(x, **kwargs)
        x = self._conv2(x, **kwargs)
        x = self._norm(x, **kwargs)
        if self._config_dict['resnet']:
            x += shortcut
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
        self._upconv = tf.keras.layers.Conv2DTranspose(filters, 3, 2, name='upsampling', **conv_kwargs)
        self._conv1 = tf.keras.layers.Conv2D(filters, 3, name='conv_1', activation='relu', **conv_kwargs)
        self._conv2 = tf.keras.layers.Conv2D(filters, 3, name='conv_2', activation='relu', **conv_kwargs)
        self._norm  = tf.keras.layers.BatchNormalization(name='norm')

    def get_config(self):
        config = super(UNetUpSampler,self).get_config()
        config.update(self._config_dict)
        return config

    def call(self, inputs, **kwargs):
        x,y = inputs
        x = self._upconv(x, **kwargs)
        shortcut = x
        x = tf.keras.activations.relu(x)
        x = tf.concat([x, y], axis=-1)
        x = self._conv1(x, **kwargs)
        x = self._conv2(x, **kwargs)
        x = self._norm(x, **kwargs)
        if self._config_dict['resnet']:
            x += shortcut
        return x

class UNetEncoder(tf.keras.layers.Layer):
    def __init__(self, n_filters=16, n_layers = 4, is_resnet=False, **kwargs):
        super(UNetEncoder, self).__init__(**kwargs)
        self._config_dict = {
            'n_filters': n_filters,
            'is_resnet': is_resnet,
            'n_layers': n_layers,
        }

    def get_config(self):
        config = super(UNetEncoder, self).get_config()
        config.update(self._config_dict)
        return config

    def build(self, input_shape):
        n_filters = self._config_dict['n_filters']
        is_resnet = self._config_dict['is_resnet']
        self._stem = [
              tf.keras.layers.Conv2D(n_filters, 3, name='stem_1', padding='same', activation='relu', kernel_initializer='he_normal'),
              tf.keras.layers.BatchNormalization(name='norm'),
              ]

        self._down_stack = []
        for k in range(self._config_dict['n_layers']):
            n_filters *= 2
            self._down_stack.append(UNetDownSampler(n_filters, is_resnet, name=f'downsample_{k+1}'),)

        super(UNetEncoder,self).build(input_shape)

    def call(self, data, **kwargs):
        x = data
        for layer in self._stem:
            x=layer(data, **kwargs)
        outputs = {'0': x}
        for i, layer in enumerate(self._down_stack):
            x = layer(x, **kwargs)
            outputs.update({str(i+1): x})
        return outputs

class UNetDecoder(tf.keras.layers.Layer):
    def __init__(self, n_filters=16, n_layers=4, is_resnet=False, **kwargs):
        super(UNetDecoder, self).__init__(**kwargs)
        self._config_dict = {
            'n_filters': n_filters,
            'is_resnet': is_resnet,
            'n_layers': n_layers,
        }

    def get_config(self):
        config = super(UNetDecoder, self).get_config()
        config.update(self._config_dict)
        return config

    def build(self, input_shape):
        n_filters = self._config_dict['n_filters']
        is_resnet = self._config_dict['is_resnet']
        n_layers = self._config_dict['n_layers']
        self._up_stack = []
        n_filters *=  2 ** n_layers
        for k in range(n_layers):
            n_filters = n_filters // 2
            self._up_stack.append(UNetUpSampler(n_filters, is_resnet, name=f'upsample_{k+1}'))

        super(UNetDecoder, self).build(input_shape)

    def call(self, data, **kwargs):
        n_layers = self._config_dict['n_layers']
        x = data[str(n_layers)] #one more conv?
        outputs={str(n_layers): x}
        for k, layer in enumerate(self._up_stack):
            k_str = str(n_layers - k - 1)
            x = layer((x, data[k_str]), **kwargs)
            outputs.update({k_str:x})
        return outputs
