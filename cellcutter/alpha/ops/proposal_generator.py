import tensorflow as tf
from ..ops import *

class ProposalGenerator:
    def __init__(self, det_model, n_cls=3, eps=1.0, min_samples=4.0, min_iou = 0.1):
        self._n_cls = n_cls
        self._eps = eps
        self._min_samples = min_samples
        self._min_iou = min_iou
        self._model = det_model

    def _ensure_shape(val):
        val = np.array(val)
        val = np.broadcast_to(val, )

    def __call__(self, images, labels=None):
        h = tf.shape(images)[1]
        w = tf.shape(images)[2]
        scaling_factor = tf.cast([h,w,h,w], tf.float32)

        det_model_out = self._model((images,labels), training=False)
        offsets = det_model_out['offsets']
        weights = det_model_out['weights']
        proposals = pred_labels(offsets, weights, eps=self._eps, min_samples=self._min_samples)
        bboxes = tf.cast(bbox_of_proposals(proposals), tf.float32) / scaling_factor  # ragged tensor
        new_data = {
            'image': images,
            'proposal': proposals,
            'proposal_bboxes': bboxes,
            'class': tf.argmax(det_model_out['cls'], axis=-1),
        }

        if labels is None:
            return new_data

        gt_bboxes = tf.cast(labels['bboxes'], tf.float32) / scaling_factor # ragged tensor
        matched_bboxes, matched_indices, matched_ious, _ = ragged_box_matching(bboxes, gt_bboxes)
        new_labels = {
            'matched_bboxes': matched_bboxes,
            'matched_ious': matched_ious,
            'matched_indices': matched_indices,
        }
        new_labels.update(labels)

        return new_data, new_labels
