import tensorflow as tf
from .layers import *

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
        self._maxpool = tf.keras.layers.MaxPool2D(name='maxpool')
        self._conv1 = BatchConv2D(filters, name='conv_norm_1', **conv_kwargs)
        self._conv2 = BatchConv2D(filters, name='conv_norm_2',**conv_kwargs)

    def get_config(self):
        base_config = super(UNetDownSampler, self).get_config()
        return dict(list(base_config.items()) + list(self._config_dict.items()))

    def call(self, inputs, **kwargs):
        x = self._maxpool(inputs, **kwargs)
        x = self._conv1(x, **kwargs)
        shortcut = x
        x = self._conv2(x, **kwargs)
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
        self._conv1 = BatchConv2D(filters, name='conv_norm_1', **conv_kwargs)
        self._conv2 = BatchConv2D(filters, name='conv_norm_2',**conv_kwargs)

    def get_config(self):
        base_config = super(UNetUpSampler, self).get_config()
        return dict(list(base_config.items()) + list(self._config_dict.items()))

    def call(self, inputs, **kwargs):
        x,y = inputs
        x = self._upconv(x, **kwargs)
        x = tf.concat([x, y], axis=-1)
        x = self._conv1(x, **kwargs)
        shortcut = x
        x = self._conv2(x, **kwargs)
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
        base_config = super(UNetEncoder, self).get_config()
        return dict(list(base_config.items()) + list(self._config_dict.items()))

    def build(self, input_shape):
        n_filters = self._config_dict['n_filters']
        is_resnet = self._config_dict['is_resnet']
        conv_kwargs = {
            'padding': 'same',
            'kernel_initializer': 'he_normal',
            'activation': 'relu'
        }
        self._stem = [
                BatchConv2D(n_filters, name='stem_1', **conv_kwargs),
                BatchConv2D(n_filters, name='stem_2', **conv_kwargs),
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
        base_config = super(UNetDecoder, self).get_config()
        return dict(list(base_config.items()) + list(self._config_dict.items()))

    def build(self, input_shape):
        n_filters = self._config_dict['n_filters']
        is_resnet = self._config_dict['is_resnet']
        n_layers = self._config_dict['n_layers']
        #self._conv1 = BatchConv2D(n_filters * 16, name='conv1')
        #self._conv2 = BatchConv2D(n_filters * 16, name='conv2')
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
