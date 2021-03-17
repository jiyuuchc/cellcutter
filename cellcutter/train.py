import tensorflow as tf
import numpy as np

def pad_cropped(cropped, offsets, size):
    '''
    pad the 'cropped' img so that the size increases by 'size'.
    padding is asymetric, controlled by 'offset' (1x2 array)
    '''
    paddings = tf.constant([[offsets[0], size - offset[0]], [offsets[1], size - offsets[1]]])
    return tf.pad(cropped, paddings)

def cutter_loss(y, coords, area_size = 320, lam = 1.0):
  bn, d0, d1 = y.shape

  y_pred = tf.math.sigmoid(y)

  loss = - tf.math.reduce_sum(y_pred)

  y_pred = tf.stack(
    [pad_cropped(cropped, coord, area_size) for cropped,coord in zip(tf.unstack(y_pred), list(coords))]
  )

  log_y_i = tf.math.log(1.0 - y_pred + tf.keras.backend.epsilon())
  log_y_i = tf.reduce_sum(log_y_i, axis = 0) - log_y_i

  loss -= tf.reduce_sum(y_pred * log_y_i)

  return loss / bn
