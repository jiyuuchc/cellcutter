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

def decode_offsets(ofs: tf.Tensor):
    _, h, w, _ = ofs.get_shape()
    x,y = tf.meshgrid(tf.range(w), tf.range(h))
    yx = tf.stack([y,x], axis=-1)
    yx = tf.cast(yx, tf.float32)
    return yx - ofs

def bbox_of_proposals(proposals):
    def _bbox_of_preds(preds_one_img): # for one image
        def np_func(img):
          h,w = img.shape
          bbox = np.array([r.bbox for r in regionprops(img + 1)], np.int32)
          return bbox if bbox.size > 0 else bbox.reshape(0,4)
        return tf.numpy_function(np_func, [preds_one_img], tf.int32)

    return tf.map_fn(
        _bbox_of_preds,
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

def _bbox_of_masks_one_image(mi_one_img):
    mi = mi_one_img.numpy()
    mi_split = np.split(mi[:,1:3], np.where(np.diff(mi[:,0]))[0]+1)
    bbox = [ind.min(axis=0).tolist() + ind.max(axis=0).tolist() for ind in mi_split]
    return tf.constant(bbox)

@tf.function(input_signature=[tf.RaggedTensorSpec((None,None,3), tf.int32, 1)])
def bbox_of_masks(mi):
    return tf.map_fn(
        lambda x : tf.py_function(_bbox_of_masks_one_image, [x], tf.int32),
        mi,
        fn_output_signature=tf.RaggedTensorSpec((None, 4), tf.int32, 0)
    )

def crop_features(feature: tf.Tensor, bboxes: tf.RaggedTensor, crop_size: int):
    return tf.image.crop_and_resize(
        feature, bboxes.merge_dims(0,1),
        tf.repeat(tf.range(bboxes.nrows(), dtype=tf.int32), bboxes.row_lengths()),
        [crop_size, crop_size]
    )

def ious_of_masks(mi_a, mi_b, h=544, w=704):
    n_mask_a = mi_a[-1,0] + 1
    n_mask_b = mi_b[-1,0] + 1
    mask_stack_a = tf.scatter_nd(mi_a, tf.ones((tf.shape(mi_a)[0],), tf.bool), [n_mask_a, h, w])
    mask_stack_b = tf.scatter_nd(mi_b, tf.ones((tf.shape(mi_b)[0],), tf.bool), [n_mask_b, h, w])
    #mask_areas_a = tf.math.count_nonzero(mask_stack_a, axis=(1,2))
    mask_areas_b = tf.math.count_nonzero(mask_stack_b, axis=(1,2))

    def iou_one_row(one_mask):
        intersects = tf.math.count_nonzero(one_mask & mask_stack_b, axis=(1,2))
        unions = tf.math.count_nonzero(one_mask) + mask_areas_b - intersects
        return tf.cast(intersects, tf.float32) / (tf.cast(unions, tf.float32) + 1.0e-7)
    ious = tf.map_fn(
        iou_one_row,
        mask_stack_a,
        fn_output_signature=tf.TensorSpec((None,), tf.float32),
    )
    return ious

def masks_to_label_img(mi, h=544, w=704):
    n_masks = mi[-1,0] + 1
    mask_stack = tf.scatter_nd(mi, tf.ones((tf.shape(mi)[0],), tf.int32), [n_masks, h, w])
    l = tf.range(n_masks, dtype=tf.int32)[::-1] + 1
    label_img = tf.reduce_max(mask_stack * l[:,None,None], axis=0)
    return label_img - 1

def crop_proposals(proposals: tf.Tensor, bboxes: tf.RaggedTensor, crop_size: int):
    label2indicator = lambda x: tf.one_hot(x, tf.reduce_max(x)+1, axis=0, dtype=tf.uint8)
    _, h, w = proposals.get_shape()
    proposal_indicator = tf.map_fn(
        label2indicator, proposals, fn_output_signature=tf.RaggedTensorSpec((None, h, w), tf.uint8, 0),
    )
    return tf.image.crop_and_resize(
        tf.expand_dims(proposal_indicator.merge_dims(0,1), -1),
        bboxes.merge_dims(0,1),
        tf.range(tf.reduce_sum(bboxes.row_lengths()), dtype=tf.int32),
        [crop_size, crop_size]
    )

def crop_masks(mask_indices: tf.RaggedTensor, bboxes: tf.RaggedTensor, matched_indices: tf.RaggedTensor, crop_size: int, h: int, w: int):
    def get_masks(inputs):
        unbatched_mask_indices, matched_indices = inputs
        masks = tf.scatter_nd(
            unbatched_mask_indices,
            tf.ones(shape=(tf.shape(unbatched_mask_indices)[0],), dtype=tf.uint8),
            shape=(unbatched_mask_indices[-1,0] + 1, h, w)
        )
        masks = tf.expand_dims(masks,-1)
        masks = tf.gather(masks, matched_indices)
        return masks
    matched_indices = tf.clip_by_value(matched_indices, 0, 999999)
    all_masks = tf.map_fn(
        get_masks, [mask_indices, matched_indices],
        fn_output_signature=tf.RaggedTensorSpec((None, h, w, 1), tf.uint8 , 0),
    )
    return tf.image.crop_and_resize(
        all_masks.merge_dims(0,1),
        bboxes.merge_dims(0,1),
        tf.range(tf.reduce_sum(bboxes.row_lengths()), dtype=tf.int32),
        [crop_size, crop_size]
    )

def mask_indices_to_image(mask_indices: tf.RaggedTensor, h:int=544, w:int=704):
    def one_frame(mi):
        n_masks = mi[-1,0] + 1
        img = tf.reduce_sum(tf.scatter_nd(mi, tf.ones(tf.shape(mi)[0], tf.uint8), (n_masks,h,w)), axis=0)
        return img
    return tf.map_fn(one_frame, mask_indices, fn_output_signature=tf.TensorSpec((h,w), dtype=tf.uint8))
