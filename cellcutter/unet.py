import tensorflow as tf
import numpy as np

class EncoderBlock(tf.keras.layers.Layer):

  INITIALIZER = 'he_normal'
  KERNEL_SIZE = 3

  def __init__(self, out_channels, with_maxpool = True, **kwargs):
    super(EncoderBlock, self).__init__(**kwargs)

    if with_maxpool:
        self.layers = [tf.keras.layers.MaxPool2D(pool_size = (2,2))]
    else:
        self.layers = []

    self.layers.append( tf.keras.layers.Conv2D(
        out_channels,
        self.KERNEL_SIZE,
        activation = 'relu',
        padding = 'same',
        kernel_initializer = self.INITIALIZER) )

    self.layers.append( tf.keras.layers.Conv2D(
        out_channels,
        self.KERNEL_SIZE,
        activation = 'relu',
        padding = 'same',
        kernel_initializer =self.INITIALIZER) )

  def call(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

class DecoderBlock(tf.keras.layers.Layer):

  INITIALIZER = 'he_normal'
  KERNEL_SIZE = 3

  def __init__(self, out_channels, **kwargs):
    super(DecoderBlock, self).__init__(**kwargs)

    self.upsampling = tf.keras.layers.UpSampling2D(size = (2,2));
    self.conv_1 = tf.keras.layers.Conv2D(
        out_channels,
        self.KERNEL_SIZE,
        activation = 'relu',
        padding = 'same',
        kernel_initializer = self.INITIALIZER
        )
    self.conv_2 = tf.keras.layers.Conv2D(
        out_channels,
        self.KERNEL_SIZE,
        activation = 'relu',
        padding = 'same',
        kernel_initializer = self.INITIALIZER
        )

  def call(self, x, h_input):
    x = self.upsampling(x)
    x = tf.keras.layers.concatenate([x, h_input])
    x = self.conv_1(x)
    x = self.conv_2(x)
    return x

class UNet3(tf.keras.Model):
  def __init__(self, n_channels = 32, **kwargs):
    super(UNet3, self).__init__(**kwargs)

    self.encoder1 = EncoderBlock(n_channels, with_maxpool=False)
    self.encoder2 = EncoderBlock(n_channels * 2)
    self.encoder3 = EncoderBlock(n_channels * 4)
    self.encoder4 = EncoderBlock(n_channels * 4)
    self.decoder1 = DecoderBlock(n_channels * 4)
    self.decoder2 = DecoderBlock(n_channels * 2)
    self.decoder3 = DecoderBlock(n_channels)

    # logits output
    self.output_layer = tf.keras.layers.Conv2D(1, 1, padding = 'same', kernel_initializer = "he_normal")

  def call(self, input):
    encoded1 = self.encoder1(input)
    encoded2 = self.encoder2(encoded1)
    encoded3 = self.encoder3(encoded2)
    encoded4 = self.encoder4(encoded3)
    decoded1 = self.decoder1(encoded4, encoded3)
    decoded2 = self.decoder2(decoded1, encoded2)
    decoded3 = self.decoder3(decoded2, encoded1)
    return self.output_layer(decoded3)

class UNet4(tf.keras.Model):
  def __init__(self, n_channels = 32, **kwargs):
    super(UNet4, self).__init__(**kwargs)

    self.encoder1 = EncoderBlock(n_channels, with_maxpool=False)
    self.encoder2 = EncoderBlock(n_channels * 2)
    self.encoder3 = EncoderBlock(n_channels * 4)
    self.encoder4 = EncoderBlock(n_channels * 8)
    self.encoder4 = EncoderBlock(n_channels * 8)

    self.decoder1 = DecoderBlock(n_channels * 8)
    self.decoder2 = DecoderBlock(n_channels * 4)
    self.decoder3 = DecoderBlock(n_channels * 2)
    self.decoder3 = DecoderBlock(n_channels)

    # logits output
    self.output_layer = tf.keras.layers.Conv2D(1, 1, padding = 'same', kernel_initializer = "he_normal")

  def call(self, input):
    encoded1 = self.encoder1(input)
    encoded2 = self.encoder2(encoded1)
    encoded3 = self.encoder3(encoded2)
    encoded4 = self.encoder4(encoded3)
    encoded5 = self.encoder4(encoded4)

    decoded1 = self.decoder1(encoded5, encoded4)
    decoded2 = self.decoder2(decoded1, encoded3)
    decoded3 = self.decoder3(decoded2, encoded2)
    decoded4 = self.decoder3(decoded3, encoded1)
    return self.output_layer(decoded4)
