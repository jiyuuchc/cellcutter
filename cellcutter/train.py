import tensorflow as tf
import numpy as np

def cutter_loss(y, coords, img_template, lam = 1.0):
  img_template.fill(0.0)

  y_pred = tf.math.sigmoid(y)
  y_pred_i = 1 - y_pred
  log_y_i = tf.math.log(y_pred_i)

  bn, d0, d1 = y.shape
  for i in range(bn):
    c0 = int(coords[i,0])
    c1 = int(coords[i,1])
    img_template[c0:c0 + d0, c1:c1+d1] += log_y_i[i,...]

  loss = - tf.math.reduce_mean(y_pred) * bn

  for i in range(bn):
    c0 = int(coords[i,0])
    c1 = int(coords[i,1])
    loss -= lam * tf.reduce_mean((img_template[c0:c0+d0, c1:c1+d1] - log_y_i[i,...]) * y_pred)

  return loss
