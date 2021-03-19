import tensorflow as tf
import numpy as np

def cutter_loss(y, coords, area_size, lam = 1.0):
  '''
  Standard loss function
  '''
  bn, d0, d1 = y.shape

  y_pred = tf.math.sigmoid(y)
  log_yi = tf.math.log(1.0 - y_pred + tf.keras.backend.epsilon())

  def pad_img(cropped, offsets):
    c0,c1 = offsets
    paddings = tf.constant([[c0, area_size - c0], [c1, area_size - c1]])
    return tf.pad(cropped, paddings)

  log_yi_padded = tf.stack(
    [pad_img(cropped, coord) for cropped,coord in zip(tf.unstack(log_yi), list(coords))]
  )
  log_yi_sum = tf.reduce_sum(log_yi_padded, axis = 0)

  log_yi  -= tf.stack(
    [log_yi_sum[c[0]:c[0]+d0, c[1]:c[1]+d1] for c in list(coords)]
  )

  loss = - tf.math.reduce_sum(y_pred)
  loss += tf.reduce_sum(y_pred * log_yi)

  return loss / bn

def cutter_loss_sm_area(y, coords, area_size, lam = 1.0):
  '''
  A loss function that might be more suitable for small areas
  '''

  bn, d0, d1 = y.shape

  y_pred = tf.math.sigmoid(y)

  def pad_img(cropped, offsets):
    c0,c1 = offsets
    paddings = tf.constant([[c0, area_size - c0], [c1, area_size - c1]])
    return tf.pad(cropped, paddings)

  y_pred_padded = tf.stack(
    [pad_img(cropped, coord) for cropped,coord in zip(tf.unstack(y_pred), list(coords))]
  )

  log_yi = tf.math.log(1.0 - y_pred + tf.keras.backend.epsilon())
  log_yi -= tf.reduce_sum(log_yi, axis = 0)

  loss = - tf.math.reduce_sum(y_pred)
  loss += tf.reduce_sum(y_pred * log_yi)

  return loss / bn
