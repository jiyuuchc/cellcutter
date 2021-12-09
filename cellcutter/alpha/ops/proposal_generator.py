from sklearn.cluster import DBSCAN
from skimage.measure import regionprops
import tensorflow as tf
from .common import *
from .boxes import *
from .clustering import *
from .box_matcher import *

# def _box_and_score(offset_imgs, weight_imgs, size_imgs, eps, min_s, min_w):
#     def np_func(l, w, s):
#         rp = regionprops(l, np.concatenate([w, s * w], axis=-1))
#         bbox = np.array([r.bbox for r in rp], np.float32).reshape(-1, 2, 2)
#         if bbox.size == 0:
#             return np.zeros((0,4), np.float32), np.zeros((0,), np.float32)
#         center = bbox.mean(axis=1, dtype=np.float32)
#         mean_values = np.array([r.mean_intensity for r in rp], np.float32)
#         area = np.array([r.area for r in rp], dtype=np.float32)
#         score = mean_values[:,0] * area
#         size = mean_values[:,1:3] / mean_values[:,0:1]
#         box = np.concatenate([center, size], axis=-1)
#         return box,score
#
#     label_imgs = pred_labels(
#         offset_imgs, weight_imgs,
#         eps=eps, min_samples=min_s, min_weight=min_w,
#         from_logits=False,
#     )
#     return tf.map_fn(
#         lambda x: tf.numpy_function(np_func, x, [tf.float32, tf.float32]),
#         [label_imgs, weight_imgs, size_imgs],
#         [tf.RaggedTensorSpec((None, 4), tf.float32, 0), tf.RaggedTensorSpec((None,), tf.float32, 0)],
#     )

def _box_and_score(offset_imgs, weight_imgs, size_imgs, min_w):
    _, height, width, _ = weight_imgs.get_shape()
    #w, indices = tf.math.top_k(tf.reshape(weight_imgs, [-1, height*width]), 2000)
    w = tf.reshape(weight_imgs, [-1, height*width])
    offset_imgs = decode_offsets(offset_imgs)
    box_imgs = tf.concat([offset_imgs, size_imgs], axis=-1)
    #boxes = tf.gather(tf.reshape(box_imgs, [-1, height*width, 4]), indices, batch_dims=1)
    boxes = tf.reshape(box_imgs, [-1, height*width,4])
    def one_frame(inputs):
        in1, in2, in3 = inputs
        ind = tf.where(in1 > in3)[:,0]
        return tf.stop_gradient(tf.gather(in2, ind)), tf.stop_gradient(tf.gather(in1, ind))
    return tf.map_fn(
        one_frame, [w, tf.cast(boxes, tf.float32), min_w],
        fn_output_signature=(tf.RaggedTensorSpec((None,4),tf.float32,0), tf.RaggedTensorSpec((None,),tf.float32,0))
    )

class ProposalGenerator:
    def __init__(self, n_cls=3, eps=1.0, min_samples=4.0, min_weight=0.01, iou_threshold=.5):
        self._n_cls = n_cls
        self._eps = tf.broadcast_to(eps, (n_cls,))
        self._min_samples = tf.broadcast_to(min_samples, (n_cls,))
        self._min_weights = tf.broadcast_to(min_weight, (n_cls,))
        self._iou_threshold = iou_threshold

    def __call__(self, labels, model_out):
        offsets = model_out['offsets']
        weights = tf.sigmoid(model_out['weights'])
        sizes = model_out['sizes']
        _, h, w, _= offsets.get_shape()

        if labels is not None:
            cls = labels['class']
        else:
            cls = tf.argmax(model_out['cls'], axis=-1, output_type=tf.int32)

        bboxes, scores = _box_and_score(
            offsets, weights, sizes,
            tf.gather(self._min_weights, cls),
        )
        # bboxes, scores = _box_and_score(
        #     offsets, weights, sizes,
        #     tf.gather(self._eps, cls),
        #     tf.gather(self._min_samples, cls),
        #     tf.gather(self._min_weights, cls),
        # )
        bboxes = box_decode(bboxes.values/[h,w,h,w])
        bboxes = tf.RaggedTensor.from_row_starts(bboxes, scores.row_starts())

        def one_frame(inputs):
            b,s = inputs
            selected_indices = tf.image.non_max_suppression(b, s, 2000, iou_threshold=self._iou_threshold)
            return tf.gather(b, selected_indices), tf.gather(s, selected_indices)
        bboxes, scores = tf.map_fn(
            one_frame, [bboxes, scores],
            fn_output_signature=(tf.RaggedTensorSpec((None,4),tf.float32,0), tf.RaggedTensorSpec((None,), tf.float32, 0)),
        )

        return bboxes, scores

        # new_data = {
        #     'proposal_bboxes': bboxes,
        #     'proposal_scores': scores,
        # }
        #
        # if labels is None:
        #     return new_data
        #
        # _,oh,ow,_ = labels['dist_map'].get_shape()
        # gt_bboxes = tf.cast(labels['bboxes'], tf.float32) / [oh, ow, oh, ow] # ragged tensor
        # matched_bboxes, matched_indices, matched_ious, _ = ragged_box_matching(bboxes, gt_bboxes)
        # new_data.update({
        #     'matched_bboxes': matched_bboxes,
        #     'matched_ious': matched_ious,
        #     'matched_indices': matched_indices,
        # })
        # return new_data
