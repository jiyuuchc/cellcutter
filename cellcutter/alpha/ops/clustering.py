import tensorflow as tf
import numpy as np
from sklearn.cluster import DBSCAN
from .common import *

def _pred_labels(locations, weights, eps, min_samples, min_weight):
    ''' generating mask proposals using dbscan
    '''
    all_labels = []
    n_batches = tf.shape(locations)[0]
    eps = tf.broadcast_to(tf.constant(eps), (n_batches,)).numpy()
    min_samples = tf.broadcast_to(tf.constant(min_samples), (n_batches,)).numpy()
    min_weight = tf.broadcast_to(tf.constant(min_weight), (n_batches,)).numpy()
    weights = weights.numpy()
    locations = locations.numpy()
    for k in range(n_batches):
        sel = weights[k] > min_weight[k]
        labels = np.ones_like(weights[k], dtype=np.int32)*(-1)
        if np.any(sel):
            dbscan = DBSCAN(eps[k], min_samples = min_samples[k])
            labels[sel] = dbscan.fit_predict(locations[k][sel,:], sample_weight=weights[k][sel])
        all_labels.append(labels)
    return tf.constant(all_labels, dtype=tf.int32)

def pred_labels(offsets, weights,  eps = 0.9, min_samples = 4, min_weight=.1, from_logits=True):
    _, h, w, _ = offsets.get_shape()
    locations = decode_offsets(offsets)
    if from_logits:
        weights = tf.sigmoid(weights)
    weights = weights[...,0]

    preds = tf.py_function(
        _pred_labels,
        [locations, weights, eps, min_samples, min_weight],
        tf.int32,
    )
    preds = tf.ensure_shape(preds, [None, h, w])
    return preds
