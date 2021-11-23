''' ops related to the detection proposal '''
import tensorflow as tf

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
        iou = intersects/(tf.math.count_nonzero(roi) + mask_areas - intersects)
        return iou

    fn = similarity_one_row
    elems = tf.range(tf.reduce_max(preds)+1)
    out_spec = tf.TensorSpec(shape=(None,), dtype=tf.float64)
    return tf.map_fn(fn, elems, fn_output_signature=out_spec)

@tf.function(input_signature=(tf.TensorSpec(shape=(None,None,None,2), dtype=tf.float32),))
def decode_offsets(ofs: tf.Tensor):
    h = tf.shape(ofs)[1]
    w = tf.shape(ofs)[2]
    x,y = tf.meshgrid(tf.range(w), tf.range(h))
    yx = tf.stack([y,x], axis=-1)
    yx = tf.cast(yx, tf.float32)
    return yx - ofs
