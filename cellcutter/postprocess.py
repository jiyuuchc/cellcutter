import numpy as np
import tensorflow as tf
from scipy.ndimage.morphology import binary_erosion

def draw_label(data, model, image, batch_size = 256):
  '''
  data: cellcutter.dataset object
  model: a tf NN model
  image: a 2D array to be drawn on
  '''
  label = 1
  dataset = data.tf_dataset()
  for coords, patches, *_ in dataset.batch(batch_size):
    coords = tf.unstack(coords)
    preds = tf.unstack(tf.squeeze(tf.math.sigmoid(model(patches))))
    for coord, pred in zip(coords, preds):
      c0,c1 = list(coord)
      patch = (pred.numpy() >= 0.5)* label
      d0,d1 = patch.shape
      image[c0:c0+d0,c1:c1+d1] = patch
      label += 1
  return image

def draw_border(data, model, image, batch_size = 256):
  '''
  data: cellcutter.dataset object
  model: a tf NN model
  image: a 2D array to be drawn on
  '''
  dataset = data.tf_dataset()
  for coords, patches, *_ in dataset.batch(batch_size):
    coords = tf.unstack(coords)
    preds = tf.unstack(tf.squeeze(tf.math.sigmoid(model(patches))))
    for coord, pred in zip(coords, preds):
      c0,c1 = list(coord)
      patch = (pred.numpy() >= 0.5).astype(np.uint8)
      edge = patch - binary_erosion(patch)
      d0,d1 = patch.shape
      image[c0:c0+d0,c1:c1+d1] += edge
  return image
