import tensorflow as tf
import tensorflow.keras.layers as layers
from .unet import UNetDownSampler
from .common import *

def _shuffle_xy(xy):
    _, height, width, channels = xy.get_shape()
    xy_split = tf.stack(tf.split(xy, num_or_size_splits=2, axis=-1), axis=-1)
    return tf.reshape(xy_split, [-1, height, width, channels])

class ShuffleDownUnit(tf.keras.layers.Layer):
    def __init__(self, n_channels, **kwargs):
        super(ShuffleDownUnit, self).__init__(**kwargs)
        self._config_dict = {
            'n_channels': n_channels,
        }
        self._block_y = [
            layers.Conv2D(n_channels//2, 1, activation='relu', name='cy1'),
            layers.BatchNormalization(name='cy1_norm'),
            layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='SAME', name='cy2'),
            layers.BatchNormalization(name='cy2_norm'),
            layers.Conv2D(n_channels//2, 1, activation='relu', name='cy3'),
            layers.BatchNormalization(name='cy3_norm'),
        ]
        self._block_x = [
            layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='SAME', name='cx1'),
            layers.BatchNormalization(name='cx1_norm'),
            layers.Conv2D(n_channels//2, 1, activation='relu', name='cx2'),
            layers.BatchNormalization(name='cx2_norm'),
        ]

    def get_config(self):
        config = super(ShuffleDownUnit, self).get_config()
        config.update(self._config_dict)
        return config

    def call(self, xy, **kwargs):
        y = xy
        for layer in self._block_y:
            y = layer(y)
        x = xy
        for layer in self._block_x:
            x = layer(x)
        xy = tf.concat([x,y], axis=-1)
        return _shuffle_xy(xy)

class ShuffleUnit(tf.keras.layers.Layer):
    def __init__(self, n_channels, **kwargs):
        super(ShuffleUnit, self).__init__(**kwargs)
        self._config_dict = {
            'n_channels': n_channels,
        }
        block=[]
        block.append(layers.Conv2D(n_channels//2, 1, activation='relu', name='c1'))
        block.append(layers.BatchNormalization(name='c1_norm'))
        block.append(layers.DepthwiseConv2D(kernel_size=3, padding='SAME', name='c2_conv'))
        block.append(layers.BatchNormalization(name='c2_norm'))
        block.append(layers.Conv2D(n_channels//2, 1, activation='relu', name='c3'))
        block.append(layers.BatchNormalization(name='c3_norm'))
        self._block = block

    def get_config(self):
        config = super(ShuffleUnit, self).get_config()
        config.update(self._config_dict)
        return config

    def call(self, xy, **kwargs):
        x, y = tf.split(xy, num_or_size_splits=2, axis=-1)
        for layer in self._block:
            y = layer(y, **kwargs)
        xy = tf.concat([x,y], axis=-1)
        return _shuffle_xy(xy)

class ShuffleBlock(tf.keras.layers.Layer):
    def __init__(self, n_units, n_channels, **kwargs):
        super(ShuffleBlock, self).__init__(**kwargs)
        self._config_dict = {
            'n_units': n_units,
            'n_channels': n_channels,
        }

        self._stack = [ShuffleDownUnit(n_channels, name='down')]
        for k in range(n_units):
            self._stack.append(ShuffleUnit(n_channels, name=f'shuffle_{k+1}'))

    def get_config(self):
        config = super(ShuffleBlock, self).get_config()
        config.update(self._config_dict)
        return config

    def call(self, xy, **kwargs):
        for layer in self._stack:
            xy = layer(xy)
        return xy

class ShuffleNetV2(tf.keras.layers.Layer):
    shuffle_net_configs = {
        '0.5x': (24, [(3, 48), (7, 96), (3, 192)]),
        '1x': (24, [(3, 116), (7, 232), (3, 464)]),
        '1.5x': (24, [(3, 176), (7, 352), (3, 704)]),
        '2x': (24, [(3, 244), (7, 488), (3, 976)]),
    }

    def __init__(self, config_key, **kwargs):
        super(ShuffleNetV2, self).__init__(**kwargs)
        self._config_dict = {
            'config_key': config_key,
        }
        net_configs = self.shuffle_net_configs[config_key]
        stem_channels = net_configs[0]
        self._stem1 = layers.Conv2D(stem_channels, 3, strides=2, activation='relu', padding='same', name='stem_conv1')
        self._norm1 = layers.BatchNormalization(name='stem1norm')
        self._stem2 = tf.keras.layers.Conv2D(stem_channels * 2, 3, strides=2, activation='relu', padding='same', name='stem_conv2')
        self._norm2 = layers.BatchNormalization(name='stem2norm')

        #self._stem2 = tf.keras.layers.MaxPool2D(name='pool')

        blocks = []
        for n_units, n_channels in net_configs[1]:
            blocks.append(ShuffleBlock(n_units=n_units, n_channels=n_channels))
        self._blocks = blocks

    def get_config(self):
        config = super(ShuffleNetV2, self).get_config()
        config.update(self._config_dict)
        return config

    def call(self, x, **kwargs):
        outputs = {'0': x}
        x = self._stem1(x)
        x = self._norm1(x)
        outputs['1'] = x
        x = self._stem2(x)
        x = self._norm2(x)
        outputs['2'] = x

        for k, shuffle_block in enumerate(self._blocks):
            x = shuffle_block(x)
            outputs[str(k+3)] = x
        return outputs
