import tensorflow as tf
import numpy as np

def cutter_loss(y, coords, area_shape, mask = None, lam = 1.0):
  '''
  Standard loss function. y is a tensor. The rest are numpy arrays.
  '''
  # area_shape is array; patch_shape is tuple
  area_shape = np.broadcast_to(area_shape, (coords.shape[-1],)) # in case area_shape is a number
  patch_shape = tuple(y.shape[1:])

  ind0 = tf.constant(list(np.ndindex(patch_shape)), dtype = tf.dtypes.int64)

  ind = ind0 + tf.constant(coords, dtype = tf.dtypes.int64)[:, tf.newaxis, :]
  ind = tf.reshape(ind, tuple(y.shape) + (len(patch_shape),))

  y_pred = tf.sigmoid(y)
  log_yi = tf.math.log(tf.clip_by_value(1.0 - y_pred, tf.keras.backend.epsilon(), 1.0))

  log_yi_sum = tf.scatter_nd(ind, log_yi, area_shape + patch_shape)
  if mask is not None:
    log_yi_sum += mask

  log_yi -= tf.gather_nd(log_yi_sum, ind)

  loss = - tf.reduce_sum(y_pred)
  loss += tf.reduce_sum(y_pred * log_yi) * lam

  return loss / y.shape[0]

def cutter_loss_low_mem(y, coords, area_shape, mask = None, lam = 1.0):
  '''
  loss function that uses sparse tensor to save memory
  y is a Tensor, the rest are numpy arrays
  '''
  area_shape = np.broadcast_to(area_shape, (coords.shape[-1],)) # in case area_shape is a number
  patch_shape = tuple(y.shape[1:])

  ind0 = tf.constant(list(np.ndindex(patch_shape)), dtype = tf.dtypes.int64)

  y_pred = tf.sigmoid(y)
  log_yi = tf.math.log(tf.clip_by_value(1.0 - y_pred, tf.keras.backend.epsilon(), 1.0))

  def pad_img(cropped, offsets):
    ind = ind0 + tf.constant(offsets, dtype = tf.dtypes.int64)
    sp = tf.sparse.SparseTensor(ind, tf.reshape(cropped, [-1]), tf.constant(area_shape + patch_shape, dtype = tf.dtypes.int64))
    return tf.sparse.expand_dims(sp, 0)

  log_yi_padded = tf.sparse.concat(
    0,
    [pad_img(cropped, coord) for cropped,coord in zip(tf.unstack(log_yi), list(coords))]
  )

  log_yi_sum = tf.sparse.reduce_sum(log_yi_padded, axis = 0)
  if mask is not None:
    log_yi_sum += mask

  ind = ind0 + tf.constant(coords, dtype = tf.dtypes.int64)[:, tf.newaxis, :]
  ind = tf.reshape(ind, tuple(log_yi.shape) + (len(patch_shape),))
  log_yi -= tf.gather_nd(log_yi_sum, ind)

  loss = - tf.reduce_sum(y_pred)
  loss += lam * tf.reduce_sum(y_pred * log_yi) * lam

  return loss /  y.shape[0]
