from sklearn.cluster import DBSCAN
from skimage.measure import regionprops
import tensorflow as tf
from .common import *
from .boxes import *
from .clustering import *
from .box_matcher import *

# def _guess_sizes(proposals, sizes, weights):
#     def one_frame(inputs):
#         p,s,w = inputs
#         indicator = tf.one_hot(p, tf.reduce_max(p)+1, axis=0)[..., None]
#         return tf.reduce_sum(s * indicator * w, axis=(1,2))/ tf.reduce_sum(indicator*w, axis=(1,2))
#     return tf.map_fn(
#         one_frame,
#         [proposals, sizes, weights],
#         fn_output_signature=tf.RaggedTensorSpec((None, 2), tf.float32, 0),
#     )

def _box_and_score(label_imgs, weight_imgs, size_imgs):
    def np_func(l, w, s):
        rp = regionprops(l, np.concatenate([w, s * w], axis=-1))
        bbox = np.array([r.bbox for r in rp], np.float32).reshape(-1, 2, 2)
        if bbox.size == 0:
            return np.zeros((0,4), np.float32), np.zeros((0,), np.float32)
        center = bbox.mean(axis=1, dtype=np.float32)
        mean_values = np.array([r.mean_intensity for r in rp], np.float32)
        area = np.array([r.area for r in rp], dtype=np.float32)
        score = mean_values[:,0] * area
        size = mean_values[:,1:3] / mean_values[:,0:1]
        box = np.concatenate([center, size], axis=-1)
        return box,score
    return tf.map_fn(
        lambda x: tf.numpy_function(np_func, x, [tf.float32, tf.float32]),
        [label_imgs, weight_imgs, size_imgs],
        [tf.RaggedTensorSpec((None, 4), tf.float32, 0), tf.RaggedTensorSpec((None,), tf.float32, 0)],
    )

class ProposalGenerator:
    def __init__(self, n_cls=3, eps=1.0, min_samples=4.0, min_weight=0.01, iou_threshold=.5):
        self._n_cls = n_cls
        self._eps = tf.broadcast_to(eps, (n_cls,))
        self._min_samples = tf.broadcast_to(min_samples, (n_cls,))
        self._min_weights = tf.broadcast_to(min_weight, (n_cls,))
        self._iou_threshold = iou_threshold

    def __call__(self, images, labels, model_out):
        offsets = model_out['offsets']
        weights = tf.sigmoid(model_out['weights'])
        _, h, w, _= offsets.get_shape()

        if labels is not None:
            cls = labels['class']
        else:
            cls = tf.argmax(model_out['cls'], axis=-1, output_type=tf.int32)

        proposals = pred_labels(
            offsets, weights,
            eps=tf.gather(self._eps, cls),
            min_samples=tf.gather(self._min_samples, cls),
            min_weight=tf.gather(self._min_weights, cls),
            from_logits=False,
        )
        bboxes, scores = _box_and_score(proposals, weights, model_out['sizes'])
        bboxes = box_decode(bboxes.values/[h,w,h,w])
        bboxes = tf.RaggedTensor.from_row_starts(bboxes, scores.row_starts())

        def one_frame(inputs):
            b,s = inputs
            selected_indices = tf.image.non_max_suppression(b, s, 2000, iou_threshold=self._iou_threshold)
            return tf.gather(b, selected_indices), tf.gather(s, selected_indices)
        bboxes, scores = tf.map_fn(
            one_frame,
            [bboxes, scores],
            (tf.RaggedTensorSpec((None,4),tf.float32,0), tf.RaggedTensorSpec((None,), tf.float32, 0)),
        )
        new_data = {
            'proposals': proposals,
            'proposal_bboxes': bboxes,
            'scores': scores,
        }

        if labels is None:
            return new_data

        _,oh,ow,_ = labels['dist_map'].get_shape()
        gt_bboxes = tf.cast(labels['bboxes'], tf.float32) / [oh, ow, oh, ow] # ragged tensor
        matched_bboxes, matched_indices, matched_ious, _ = ragged_box_matching(bboxes, gt_bboxes)
        new_labels = {
            'matched_bboxes': matched_bboxes,
            'matched_ious': matched_ious,
            'matched_indices': matched_indices,
        }
        return new_data, new_labels
