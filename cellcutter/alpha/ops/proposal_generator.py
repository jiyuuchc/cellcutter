from sklearn.cluster import DBSCAN
from skimage.measure import regionprops
import tensorflow as tf
from .common import *
from .boxes import *
from .clustering import *
from .box_matcher import *

def _box_and_score_dbscan(offset_imgs, weight_imgs, size_imgs, eps, min_s, min_w):
    label_imgs = pred_labels(
        offset_imgs, weight_imgs,
        eps=eps, min_samples=min_s, min_weight=min_w,
        from_logits=False,
    )

    def np_func(l, o, w, s):
        rp = regionprops(l+1, np.concatenate([w, o * w, s * w], axis=-1))
        # bbox = np.array([r.bbox for r in rp], np.float32).reshape(-1, 2, 2)
        # if bbox.size == 0:
        #     return np.zeros((0,4), np.float32), np.zeros((0,), np.float32)
        # center = bbox.mean(axis=1, dtype=np.float32)
        mean_values = np.array([r.mean_intensity for r in rp], np.float32)
        if mean_values.size == 0:
             return np.zeros((0,4), np.float32), np.zeros((0,), np.float32)
        area = np.array([r.area for r in rp], dtype=np.float32)
        score = mean_values[:,0] * area
        box = mean_values[:,1:5] / mean_values[:,0:1]
        return box,score

    return tf.map_fn(
        lambda x: tf.numpy_function(np_func, x, [tf.float32, tf.float32]),
        [label_imgs, decode_offsets(offset_imgs), weight_imgs, size_imgs],
        [tf.RaggedTensorSpec((None, 4), tf.float32, 0), tf.RaggedTensorSpec((None,), tf.float32, 0)],
    )

def _box_and_score_simple(offset_imgs, weight_imgs, size_imgs, min_w):
    _, height, width, _ = weight_imgs.get_shape()
    w = tf.reshape(weight_imgs, [-1, height*width])
    offset_imgs = decode_offsets(offset_imgs)
    box_imgs = tf.concat([offset_imgs, size_imgs], axis=-1)
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
    def __init__(self, n_cls=3, eps=1.0, min_samples=4.0, min_weight=0.01, iou_threshold=.5, use_dbscan=False):
        self._n_cls = n_cls
        self._eps = tf.broadcast_to(eps, (n_cls,))
        self._min_samples = tf.broadcast_to(min_samples, (n_cls,))
        self._min_weights = tf.broadcast_to(min_weight, (n_cls,))
        self._iou_threshold = iou_threshold
        self._use_dbscan = use_dbscan

    def __call__(self, labels, model_out):
        offsets = model_out['offsets']
        weights = tf.sigmoid(model_out['weights'])
        sizes = model_out['sizes']
        _, h, w, _= offsets.get_shape()

        if labels is not None:
            cls = labels['class']
        else:
            cls = tf.argmax(model_out['cls'], axis=-1, output_type=tf.int32)

        if self._use_dbscan:
            bboxes, scores = _box_and_score_dbscan(
                  offsets, weights, sizes,
                  tf.gather(self._eps, cls),
                  tf.gather(self._min_samples, cls),
                  tf.gather(self._min_weights, cls),
            )
        else:
            bboxes, scores = _box_and_score_simple(
                offsets, weights, sizes,
                tf.gather(self._min_weights, cls),
            )
        bboxes = box_decode(bboxes.values/[h,w,h,w])
        bboxes = tf.RaggedTensor.from_row_starts(bboxes, scores.row_starts())

        def nms(inputs):
            selected_indices = tf.image.non_max_suppression(*inputs, 2000, iou_threshold=self._iou_threshold)
            return tf.gather(inputs[0], selected_indices), tf.gather(inputs[1], selected_indices)
        bboxes, scores = tf.map_fn(
            nms, [bboxes, scores],
            fn_output_signature=(tf.RaggedTensorSpec((None,4),tf.float32,0), tf.RaggedTensorSpec((None,), tf.float32, 0)),
        )

        return bboxes, scores
