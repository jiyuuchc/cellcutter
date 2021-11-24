''' ops related to the detection proposal '''
import tensorflow as tf
import numpy as np
from skimage.measure import regionprops

@tf.function(input_signature=(
    tf.TensorSpec(shape=(None,None), dtype=tf.int32),
    tf.TensorSpec(shape=(None,3), dtype=tf.int32),)
)
def proposal_iou(preds, mi):
    h = tf.shape(preds)[0]
    w = tf.shape(preds)[1]
    n_masks = mi[-1,0] + 1
    masks = tf.scatter_nd(mi, tf.ones(shape=(tf.shape(mi)[0],), dtype=tf.bool), shape=(n_masks, h, w))
    mask_areas = tf.math.count_nonzero(masks, axis=(1,2))

    def similarity_one_row(k):
        roi = preds==k
        intersects = tf.math.count_nonzero(tf.logical_and(roi, masks), axis=(1,2))
        iou = tf.cast(intersects, tf.float32)/tf.cast(tf.math.count_nonzero(roi) + mask_areas - intersects, tf.float32)
        return iou

    fn = similarity_one_row
    elems = tf.range(tf.reduce_max(preds)+1)
    out_spec = tf.TensorSpec(shape=(None,), dtype=tf.float32)
    return tf.map_fn(fn, elems, fn_output_signature=out_spec)

@tf.function(input_signature=(tf.TensorSpec(shape=(None,None,None,2), dtype=tf.float32),))
def decode_offsets(ofs: tf.Tensor):
    h = tf.shape(ofs)[1]
    w = tf.shape(ofs)[2]
    x,y = tf.meshgrid(tf.range(w), tf.range(h))
    yx = tf.stack([y,x], axis=-1)
    yx = tf.cast(yx, tf.float32)
    return yx - ofs

def _bbox_of_preds(preds_one_img): # for one image
    h,w = preds_one_img.shape
    bbox = np.array([r.bbox for r in regionprops(preds_one_img + 1)])
    return bbox.astype(np.int32)

def bbox_of_proposals(proposals):
    return tf.map_fn(
        lambda x: tf.numpy_function(_bbox_of_preds, [x], tf.int32),
        proposals,
        fn_output_signature=tf.RaggedTensorSpec((None,4), tf.int32, 0)
    )

def _bbox_of_masks(mi_ragged):
    all_results = []
    for mi in mi_ragged:
        mi = mi.numpy()
        mi_split = np.split(mi[:,1:3], np.where(np.diff(mi[:,0]))[0]+1)
        bbox = [ind.min(axis=0).tolist() + ind.max(axis=0).tolist() for ind in mi_split]
        # y0x0 = np.array([ ind.min(axis=0) for ind in mi_split], np.int32)
        # y1x1 = np.array([ ind.max(axis=0) for ind in mi_split], np.int32)
        # bbox = np.concatenate([y0x0, y1x1], axis=-1)
        all_results.append(bbox)
    return tf.ragged.constant(all_results, ragged_rank=1)

@tf.function(input_signature=[tf.RaggedTensorSpec((None,None,3), tf.int32, 1)])
def bbox_of_masks(mi):
    return tf.py_function(_bbox_of_masks, [mi], tf.RaggedTensorSpec((None,None,None),tf.int32, 1))
