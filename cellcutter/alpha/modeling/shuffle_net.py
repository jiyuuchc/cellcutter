import tensorflow as tf
import tensorflow.keras.layers as layers
from .unet import UNetDownSampler
from .layers import *

class ShuffleUnit(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels=None, down_sampling = False, **kwargs):
        super(ShuffleUnit, self).__init__(**kwargs)
        self._config_dict = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'down_sampling': down_sampling,
        }
        strides = 2 if down_sampling else 1
        out_channels = out_channels if out_channels else in_channels
        block=[]
        block.append(BatchConv2D(in_channels, size=1, name='c1'))
        block.append(layers.DepthwiseConv2D(kernel_size=3, strides=strides, padding='SAME', name='c2_conv'))
        block.append(layers.BatchNormalization(name='c2_norm'))
        block.append(BatchConv2D(out_channels, size=1, name='c3'))
        self._layers = block

    def get_config(self):
        base_config = super(UNetDownSampler, self).get_config()
        return dict(list(base_config.items()) + list(self._config_dict.items()))

    def call(self, x, **kwargs):
        for layer in self._layers:
            x = layer(x, **kwargs)
        return x

class ShuffleBlock(tf.keras.layers.Layer):
    def __init__(self, num_units, in_channels, out_channels=None, **kwargs):
        super(ShuffleBlock, self).__init__(**kwargs)
        self._config_dict = {
            'num_units': num_units,
            'in_channels': in_channels,
            'out_channels': out_channels,
        }
        out_channels = 2 * self.in_channels if out_channels is None else out_channels

        self._branch1 = [
            layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='SAME', name='branch2_conv1'),
            layers.BatchNormalization(name='branch2_norm1'),
            layers.ReLU(name='branch2_relu1'),
            layers.Conv2D(out_channels//2, 1, padding='same', name='branch2_conv2'),
            layers.BatchNormalization(name='branch2_norm2'),
            layers.ReLU(name='branch2_relu2'),
        ]
        self._branch2 = ShuffleUnit(in_channels=in_channels, out_channels=out_channels//2, down_sampling=True, name='branch1')

        self._stack = []
        for k in range(num_units):
            self._stack.append(ShuffleUnit(in_channels=out_channels//2, name=f'shuffle_{k+1}'))

    def get_config(self):
        base_config = super(UNetDownSampler, self).get_config()
        return dict(list(base_config.items()) + list(self._config_dict.items()))

    def _shuffle_xy(self, x, y):
        batch_size, height, width, channels = x.shape[:]
        depth = channels
        z = tf.stack([x, y], axis=-1)
        z = tf.transpose(z, [0, 1, 2, 4, 3])
        z = tf.reshape(z, [-1, height, width, 2*depth])
        x, y = tf.split(z, num_or_size_splits=2, axis=3)
        return x, y

    def call(self, x, **kwargs):
        branch2 = self._branch2(x, **kwargs)
        for layer in self._branch1:
            x = layer(x, **kwargs)
        branch1 = x

        basic_uint_count = 0
        for shuffle_unit in self._stack:
            branch1, branch2 = self._shuffle_xy(branch1, branch2)
            branch1 = shuffle_unit(branch1, **kwargs)

        return layers.concatenate([branch1, branch2])

class ShuffleNetV2(tf.keras.layers.Layer):
    shuffle_net_configs = {
        '0.5x': (24, [(3, 48), (7, 96), (3, 192)]),
        '1x': (24, [(3, 116), (7, 232), (3, 464)]),
        '1.5x': (24, [(3, 176), (7, 352), (3, 704)]),
        '2x': (24, [(3, 244), (7, 488), (3, 976)]),
    }

    def __init__(self, config_key, **kwargs):
        super(ShuffleNetV2, self).__init__(**kwargs)

        net_configs = self.shuffle_net_configs[config_key]
        stem_channels = net_configs[0]
        self._stem1 = BatchConv2D(stem_channels, name='stem1')
        self._stem2 = UNetDownSampler(stem_channels, name='stem2')
        self._stem3 = UNetDownSampler(stem_channels*2, name='stem3')

        blocks = []
        in_channels = stem_channels
        for num_units, out_channels in net_configs[1]:
            blocks.append(ShuffleBlock(num_units=num_units, in_channels=in_channels, out_channels=out_channels))
            in_channels = out_channels
        self._blocks = blocks

    def call(self, x, **kwargs):
        x = self._stem1(x)
        outputs = {'0': x}
        x = self._stem2(x)
        outputs['1'] = x
        x = self._stem3(x)
        outputs['2'] = x

        for k, shuffle_block in enumerate(self._blocks):
            x = shuffle_block(x)
            outputs[str(k+3)] = x
        return outputs
