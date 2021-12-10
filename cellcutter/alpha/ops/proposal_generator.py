from sklearn.cluster import DBSCAN
from skimage.measure import regionprops
import tensorflow as tf
from .common import *
from .boxes import *
from .clustering import *
from .box_matcher import *

def _get_top_k(weight_imgs, offset_imgs, size_imgs, topk):
    _, height, width, _ = weight_imgs.get_shape()
    w = tf.reshape(weight_imgs, [-1, height*width])
    scores, indices = tf.math.top_k(w, k=topk)
    bbox_imgs = tf.concat([offset_imgs, size_imgs], axis=-1)
    bboxes = tf.reshape(bbox_imgs, [-1, height*width, 4])
    bboxes = tf.gather(bboxes, indices, batch_dims=1)
    return bboxes, scores

class ProposalGenerator:
    def __init__(self, n_cls=3, eps=1.0, min_weight=0.1, iou_threshold=.5, use_dbscan=False):
        self._n_cls = n_cls
        self._eps = tf.broadcast_to(eps, (n_cls,))
        self._min_weights = tf.broadcast_to(min_weight, (n_cls,))
        self._iou_threshold = iou_threshold
        self._use_dbscan = use_dbscan

    def _box_and_score_dbscan(self, offset_imgs, weight_imgs, size_imgs, epses, min_weights, topk=10000):
        batch_bboxes, batch_scores = _get_top_k(weight_imgs, offset_imgs, size_imgs, topk)
        def np_func(bboxes, scores, eps, min_weight):
            dbscan = DBSCAN(eps=eps, min_samples=min_weight)
            pred_labels = dbscan.fit_predict(bboxes, sample_weight=scores)
            pred_labels = pred_labels.reshape(-1,1) + 1
            intensity_img = np.concatenate([scores[...,None],bboxes],axis=-1)
            rp = regionprops(pred_labels, intensity_image=intensity_img.reshape(-1,1,5))
            mean_values = np.array([r.mean_intensity for r in rp], np.float32)
            if mean_values.size == 0:
                 return np.zeros((0,4), np.float32), np.zeros((0,), np.float32)
            area = np.array([r.area for r in rp], dtype=np.float32)
            new_scores = mean_values[:,0] * area
            new_bboxes = mean_values[:, 1:5] / mean_values[:,0:1]
            return new_bboxes, new_scores
        boxes,scores= tf.map_fn(
            lambda x: tf.numpy_function(np_func, x, [tf.float32, tf.float32]),
            (batch_bboxes, batch_scores, epses,  min_weights),
            fn_output_signature=[tf.RaggedTensorSpec((None, 4), tf.float32, 0), tf.RaggedTensorSpec((None,), tf.float32, 0)],
        )
        boxes = box_decode(boxes.values)
        boxes = tf.RaggedTensor.from_row_lengths(boxes, scores.row_lengths())
        return boxes, scores

    def _box_and_score_simple(self, offset_imgs, weight_imgs, size_imgs, min_w, topk=50000):
        bboxes, scores = _get_top_k(weight_imgs, offset_imgs, size_imgs, topk)
        bboxes = box_decode(bboxes)
        def nms(inputs):
            input_boxes, input_scores, score_threshold = inputs
            selected_indices = tf.image.non_max_suppression(
                  input_boxes, input_scores, 2000,
                  iou_threshold=self._iou_threshold,
                  score_threshold=score_threshold,
            )
            return (
                tf.stop_gradient(tf.gather(input_boxes, selected_indices)),
                tf.stop_gradient(tf.gather(input_scores, selected_indices)),
            )
        return tf.map_fn(
            nms, [bboxes, scores, min_w],
            fn_output_signature=(tf.RaggedTensorSpec((None,4),tf.float32,0), tf.RaggedTensorSpec((None,), tf.float32, 0)),
        )
        # def one_frame(inputs):
        #     in1, in2, in3 = inputs
        #     ind = tf.where(in1 > in3)[:,0]
        #     return tf.stop_gradient(tf.gather(in2, ind)), tf.stop_gradient(tf.gather(in1, ind))
        # return tf.map_fn(
        #     one_frame, [w, tf.cast(boxes, tf.float32), min_w],
        #     fn_output_signature=(tf.RaggedTensorSpec((None,4),tf.float32,0), tf.RaggedTensorSpec((None,),tf.float32,0))
        # )

    def __call__(self, labels, model_out):
        offsets = decode_offsets(model_out['offsets'])
        weights = tf.sigmoid(model_out['weights'])
        sizes = model_out['sizes']
        _, h, w, _= offsets.get_shape()

        if labels is not None:
            cls = labels['class']
        else:
            cls = tf.argmax(model_out['cls'], axis=-1, output_type=tf.int32)

        if self._use_dbscan:
            bboxes, scores = self._box_and_score_dbscan(
                  offsets, weights, sizes,
                  tf.gather(self._eps, cls),
                  tf.gather(self._min_weights, cls),
            )
        else:
            bboxes, scores = self._box_and_score_simple(
                offsets, weights, sizes,
                tf.gather(self._min_weights, cls),
            )
        bboxes = bboxes / [h,w,h,w]
        return bboxes, scores
