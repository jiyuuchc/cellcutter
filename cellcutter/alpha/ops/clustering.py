import tensorflow as tf
import numpy as np
from sklearn.cluster import DBSCAN
from .common import *

def _pred_labels(locations, weights, eps, min_samples, min_weight):
    ''' generating mask proposals using dbscan
    '''
    all_labels = []
    dbscan = DBSCAN(eps, min_samples = min_samples)
    for loc, w in zip(locations.numpy(), weights.numpy()):
        labels = np.ones_like(w, dtype=np.int32)*(-1)
        sel = w > min_weight
        labels[sel] = dbscan.fit_predict(loc[sel,:], sample_weight=w[sel])
        all_labels.append(labels)
    return tf.constant(all_labels, dtype=tf.int32)

def pred_labels(offsets, weights,  from_logits=True, eps = 0.9, min_samples = 4, min_weight=.1):
    locations = decode_offsets(offsets)
    if from_logits:
        weights = tf.sigmoid(weights)
    weights = weights[...,0]

    preds = tf.py_function(
        _pred_labels,
        [locations, weights, eps, min_samples, min_weight],
        tf.int32,
    )
    return preds
