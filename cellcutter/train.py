import tensorflow as tf
import numpy as np

def augment(img, t):
  if t == 1 or t == 3:
    img = tf.image.flip_left_right(img)
  if t == 2 or t == 3:
    img = tf.image.flip_up_down(img)
  return img
