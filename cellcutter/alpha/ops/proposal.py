''' ops related to the detection proposal '''
import tensorflow as tf

@tf.function(
    input_signiture=[
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
    ],
)
def proposal_ious(proposals, masks):
    ''' compute iou between proposal masks and true masks
    Inputs:
      proposals:
          HxW tensor representing proposals as labels (0-indexed). label==-1 indicating background
      masks:
          HxWxN tensor represent ground truth. Each slice is a separate mask.
    Outputs:
      MxN tensor of pairwise IOUs. M is the number of labels in proposals.
    '''
    mask_areas = tf.math.count_nonzero(masks, axis=(1,2))

    def iou_one_row(k):
        roi = preds==k
        intersects = tf.math.count_nonzero(tf.logical_and(roi, masks), axis=(1,2))
        iou = tf.cast(intersects/(tf.math.count_nonzero(roi) + mask_areas), tf.float32)
        return iou

    return tf.map_fn(
      iou_one_row,
      tf.range(tf.reduce_max(preds)+1),
      fn_output_signature=tf.TensorSpec(shape=(masks.shape[0],), dtype=tf.float32),
    )
