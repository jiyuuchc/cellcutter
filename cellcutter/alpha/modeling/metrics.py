import tensorflow as tf
from ..ops import *

class LabelMaskIou(tf.keras.metrics.Metric):
    def __init__(self, name='det_model_box_iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self._r = self.add_weight(name='r50', shape=(3,), initializer='zeros')
        self._n_masks = self.add_weight(name='nmasks', initializer='zeros')
        self._n_detections = self.add_weight(name='ndetections', initializer='zeros')

    def update_state(self, labels, mask_indices):
        matched_indices, matched_ious = mask_matching(labels, mask_indices)

        n_masks = tf.reduce_sum(tf.gather(mask_indices.values, mask_indices.row_limits()-1)[:,0] + 1)
        self._n_masks.assign_add(tf.cast(n_masks, self.dtype))
        n_detections = tf.reduce_sum(matched_ious.row_lengths())
        self._n_detections.assign_add(tf.cast(n_detections, self.dtype))

        r_values = tf.stack([
            tf.math.count_nonzero(matched_ious.values>0.3),
            tf.math.count_nonzero(matched_ious.values>0.5),
            tf.math.count_nonzero(matched_ious.values>0.75),
        ])
        self._r.assign_add(tf.cast(r_values, self.dtype))

    def result(self):
        return tf.stack([self._r / self._n_detections, self._r / self._n_masks], axis=0)

class LabelBoxIou(tf.keras.metrics.Metric):
    def __init__(self, name='label_box_iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self._r = self.add_weight(name='r50', shape=(3,), initializer='zeros')
        self._n_masks = self.add_weight(name='nmasks', initializer='zeros')
        self._n_detections = self.add_weight(name='ndetections', initializer='zeros')

    def update_state(self, labels, boxes):
        bboxes = tf.cast(bbox_of_proposals(labels), tf.float32) # ragged tensor
        gt_bboxes = tf.cast(boxes, tf.float32) #ragged tensor

        _, matched_indices, matched_ious, _ = ragged_box_matching(bboxes, gt_bboxes)
        n_gt_masks = tf.reduce_sum(gt_bboxes.row_lengths())
        n_detections = tf.reduce_sum(bboxes.row_lengths())
        self._n_masks.assign_add(tf.cast(n_gt_masks, self.dtype))
        self._n_detections.assign_add(tf.cast(n_detections, self.dtype))

        r_values = tf.stack([
            tf.math.count_nonzero(matched_ious.values>0.3),
            tf.math.count_nonzero(matched_ious.values>0.5),
            tf.math.count_nonzero(matched_ious.values>0.75),
        ])
        self._r.assign_add(tf.cast(r_values, self.dtype))

    def result(self):
        return tf.stack([self._r / self._n_detections, self._r / self._n_masks], axis=0)
