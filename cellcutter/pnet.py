import tensorflow as tf
import numpy as np

__all__ = ['PNet4', 'PNet5']

class DilatedConvolutionBlock(tf.keras.layers.Layer):

  INITIALIZER = 'he_normal'
  KERNEL_SIZE = 3

  def __init__(self, out_channels, dilation, with_batch_normalization = True, **kwargs):
    super(DilatedConvolutionBlock, self).__init__(**kwargs)

    self.layers.append( tf.keras.layers.Conv2D(
      out_channels,
      self.KERNEL_SIZE,
      dilation_rate = dilation,
      activation = 'relu',
      padding = 'same',
      kernel_initializer = self.INITIALIZER,
      name = 'conv_a') )

    self.layers.append( tf.keras.layers.Conv2D(
      out_channels,
      self.KERNEL_SIZE,
      dilation_rate = dilation,
      activation = 'relu',
      padding = 'same',
      kernel_initializer =self.INITIALIZER,
      name = 'conv_b') )

    if with_batch_normalization:
      self.layers.append( tf.kerea.layers.BatchNormalization(name = 'bn'))

  def call(self, x, **kwargs):
    for layer in self.layers:
      x = layer(x, **kwargs)
    return x

class PNet4(tf.keras.Model):
  def __init__(self, n_channels = 32, **kwargs):
    super(UNet3, self).__init__(**kwargs)

    self.conv_block1 = DilatedConvolutionBlock(n_channels, 1, name='block1')
    self.conv_block2 = DilatedConvolutionBlock(n_channels, 2, name='block2')
    self.conv_block3 = DilatedConvolutionBlock(n_channels, 4, name='block3')
    self.conv_block4 = DilatedConvolutionBlock(n_channels, 8, name='block4')
    self.concat = tf.keras.layers.Concatenate(name = 'concat')
    self.interp1 = tf.keras.layers.Conv2D(
      n_channels * 4,
      1,
      activation = 'relu',
      padding = 'same',
      kernel_initializer =self.INITIALIZER,
      name = 'interp1')
    self.interp2 = tf.keras.layers.Conv2D(
      n_channels * 4,
      1,
      activation = 'relu',
      padding = 'same',
      kernel_initializer =self.INITIALIZER,
      name = 'interp2')
    self.output = tf.keras.layers.Conv2D(
      1,
      1,
      activation = 'relu',
      padding = 'same',
      kernel_initializer =self.INITIALIZER,
      name = 'output')

  def __call__(self, input, **kwargs):
    b1 = self.conv_block1(input, **kwargs)
    b2 = self.conv_block2(b1, **kwargs)
    b3 = self.conv_block3(b2, **kwargs)
    b4 = self.conv_block4(b3, **kwargs)
    concated = self.concat([b1,b2,b3,b4], **kwargs)
    i1 = self.interp1(concated, **kwargs)
    i2 = self.interp2(i1, **kwargs)
    output = self.output(i2, **kwargs)
    return output

class PNet5(tf.keras.Model):
  def __init__(self, n_channels = 32, **kwargs):
    super(UNet3, self).__init__(**kwargs)

    self.conv_block1 = DilatedConvolutionBlock(n_channels, 1, name='block1')
    self.conv_block2 = DilatedConvolutionBlock(n_channels, 2, name='block2')
    self.conv_block3 = DilatedConvolutionBlock(n_channels, 4, name='block3')
    self.conv_block4 = DilatedConvolutionBlock(n_channels, 8, name='block4')
    self.conv_block5 = DilatedConvolutionBlock(n_channels, 12, name='block4')
    self.concat = tf.keras.layers.Concatenate(name = 'concat')
    self.interp1 = tf.keras.layers.Conv2D(
      n_channels * 5,
      1,
      activation = 'relu',
      padding = 'same',
      kernel_initializer =self.INITIALIZER,
      name = 'interp1')
    self.interp2 = tf.keras.layers.Conv2D(
      n_channels * 5,
      1,
      activation = 'relu',
      padding = 'same',
      kernel_initializer =self.INITIALIZER,
      name = 'interp2')
    self.output = tf.keras.layers.Conv2D(
      1,
      1,
      activation = 'relu',
      padding = 'same',
      kernel_initializer =self.INITIALIZER,
      name = 'output')

  def __call__(self, input, **kwargs):
    b1 = self.conv_block1(input, **kwargs)
    b2 = self.conv_block2(b1, **kwargs)
    b3 = self.conv_block3(b2, **kwargs)
    b4 = self.conv_block4(b3, **kwargs)
    b4 = self.conv_block5(b4, **kwargs)
    concated = self.concat([b1,b2,b3,b4,b5], **kwargs)
    i1 = self.interp1(concated, **kwargs)
    i2 = self.interp2(i1, **kwargs)
    output = self.output(i2, **kwargs)
    return output
