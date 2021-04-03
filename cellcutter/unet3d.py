import tensorflow as tf
import numpy as np

__all__ = ['VUNet3']

class EncoderBlock3D(tf.keras.layers.Layer):

  INITIALIZER = 'he_normal'
  KERNEL_SIZE = 3

  def __init__(self, out_channels, with_maxpool = True, with_batch_normalization = False, **kwargs):
    super(EncoderBlock, self).__init__(**kwargs)

    if with_maxpool:
      self.layers = [tf.keras.layers.MaxPool3D(name = 'maxpool')]
    else:
      self.layers = []

    self.layers.append( tf.keras.layers.Conv3D(
      out_channels,
      self.KERNEL_SIZE,
      activation = 'relu',
      padding = 'same',
      kernel_initializer = self.INITIALIZER,
      name = 'conv_a') )

    self.layers.append( tf.keras.layers.Conv3D(
      out_channels,
      self.KERNEL_SIZE,
      activation = 'relu',
      padding = 'same',
      kernel_initializer =self.INITIALIZER,
      name = 'conv_b') )

    if with_batch_normalization:
      self.layers.append( tf.keras.layers.BatchNormalization(name = 'bn'))

  def call(self, x, **kwargs):
    for layer in self.layers:
      x = layer(x, **kwargs)
    return x

class DecoderBlock3D(tf.keras.layers.Layer):

  INITIALIZER = 'he_normal'
  KERNEL_SIZE = 3

  def __init__(self, out_channels, with_batch_normalization = False, **kwargs):
    super(DecoderBlock, self).__init__(**kwargs)

    self.upsampling = tf.keras.layers.UpSampling3D(name = 'upsampling');
    self.conv_1 = tf.keras.layers.Conv3D(
        out_channels,
        self.KERNEL_SIZE,
        activation = 'relu',
        padding = 'same',
        kernel_initializer = self.INITIALIZER,
        name = 'conv_a'
        )
    self.conv_2 = tf.keras.layers.Conv3D(
        out_channels,
        self.KERNEL_SIZE,
        activation = 'relu',
        padding = 'same',
        kernel_initializer = self.INITIALIZER,
        name = 'conv_b'
        )
    if with_batch_normalization:
      self.bn = tf.keras.layers.BatchNormalization(name = 'bn')

  def call(self, x, h_input, **kwargs):
    x = self.upsampling(x, **kwargs)
    x = tf.keras.layers.concatenate([x, h_input])
    x = self.conv_1(x, **kwargs)
    x = self.conv_2(x, **kwargs)
    if hasattr(self, 'bn'):
      x = self.bn(x, **kwargs)
    return x

class VUNet3(tf.keras.Model):
  def __init__(self, n_channels = 32, bn = False, **kwargs):
    super(UNet3, self).__init__(**kwargs)

    self.encoder1 = EncoderBlock3D(n_channels, with_maxpool=False, with_batch_normalization = bn, name='encoder1')
    self.encoder2 = EncoderBlock3D(n_channels * 2, with_batch_normalization = bn, name='encoder2')
    self.encoder3 = EncoderBlock3D(n_channels * 4, with_batch_normalization = bn, name='encoder3')
    self.encoder4 = EncoderBlock3D(n_channels * 4, with_batch_normalization = bn, name='encoder4')
    self.decoder1 = DecoderBlock3D(n_channels * 4, with_batch_normalization = bn, name='decoder1')
    self.decoder2 = DecoderBlock3D(n_channels * 2, with_batch_normalization = bn, name='decoder2')
    self.decoder3 = DecoderBlock3D(n_channels, with_batch_normalization = bn, name='decoder3')

    # logits output
    self.output_layer = tf.keras.layers.Conv3D(1, 1, padding = 'same', kernel_initializer = "he_normal", name='output')

  def call(self, input, **kwargs):
    encoded1 = self.encoder1(input, **kwargs)
    encoded2 = self.encoder2(encoded1, **kwargs)
    encoded3 = self.encoder3(encoded2, **kwargs)
    encoded4 = self.encoder4(encoded3, **kwargs)
    decoded1 = self.decoder1(encoded4, encoded3, **kwargs)
    decoded2 = self.decoder2(decoded1, encoded2, **kwargs)
    decoded3 = self.decoder3(decoded2, encoded1, **kwargs)
    return self.output_layer(decoded3, **kwargs)
