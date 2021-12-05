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
        shortcut= x
        x = self._conv1(x, **kwargs)
        x = self._conv2(x, **kwargs)
        if self._config_dict['resnet']:
            x = tf.concat([x, shortcut], axis=-1)
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
        self._upsample = tf.keras.layers.UpSampling2D(name='upsampling')
        self._conv1 = BatchConv2D(filters, name='conv_norm_1', **conv_kwargs)
        self._conv2 = BatchConv2D(filters, name='conv_norm_2',**conv_kwargs)

    def get_config(self):
        base_config = super(UNetUpSampler, self).get_config()
        return dict(list(base_config.items()) + list(self._config_dict.items()))

    def call(self, inputs, **kwargs):
        x = self._upsample(inputs, **kwargs)
        shortcut= x
        x = self._conv1(x, **kwargs)
        x = self._conv2(x, **kwargs)
        if self._config_dict['resnet']:
            x = tf.concat([x, shortcut], axis=-1)
        return x

class UNetEncoder(tf.keras.layers.Layer):
    def __init__(self, n_filters=16, is_resnet=False, **kwargs):
        super(UNetEncoder, self).__init__(**kwargs)
        self._config_dict = {
            'n_filters': n_filters,
            'is_resnet': is_resnet,
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
        self._down_stack = [
                UNetDownSampler(n_filters * 2, is_resnet, name='downsample_1'),
                UNetDownSampler(n_filters * 4, is_resnet, name='downsample_2'),
                UNetDownSampler(n_filters * 8, is_resnet, name='downsample_3'),
                UNetDownSampler(n_filters * 16, is_resnet, name='downsample_4'),
                UNetDownSampler(n_filters * 32, is_resnet, name='downsample_5'),
                ]
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
    def __init__(self, n_filters=16, is_resnet=False, **kwargs):
        super(UNetDecoder, self).__init__(**kwargs)
        self._config_dict = {
            'n_filters': n_filters,
            'is_resnet': is_resnet,
        }

    def get_config(self):
        base_config = super(UNetDecoder, self).get_config()
        return dict(list(base_config.items()) + list(self._config_dict.items()))

    def build(self, input_shape):
        n_filters = self._config_dict['n_filters']
        is_resnet = self._config_dict['is_resnet']
        #self._conv1 = BatchConv2D(n_filters * 16, name='conv1')
        #self._conv2 = BatchConv2D(n_filters * 16, name='conv2')
        self._up_stack = [
                UNetUpSampler(n_filters * 16, is_resnet, name='upsample_1'), # x8
                UNetUpSampler(n_filters * 8, is_resnet, name='upsample_2'), # x8
                UNetUpSampler(n_filters * 4, is_resnet, name='upsample_3'), # x4
                UNetUpSampler(n_filters * 2, is_resnet, name='upsample_4'),  # x2
                UNetUpSampler(n_filters, is_resnet, name='upsample_5'),  # x1
                ]
        super(UNetDecoder, self).build(input_shape)

    def call(self, data, **kwargs):
        x = data['5'] #one more conv?
        outputs={'5': x}
        for layer, k in zip(self._up_stack, ['4','3','2','1','0']):
            x = layer(x, **kwargs)
            x = tf.concat([x, data[k]], axis = -1)
            outputs.update({k:x})
        return outputs
