import tensorflow as tf

def iou_loss(gt_mask, mask, from_logits=False):
    gt_mask = tf.cast(gt_mask, tf.float32)
    mask = tf.cast(mask, tf.float32)
    if from_logits:
        mask = tf.sigmoid(mask)
    intersec = tf.reduce_sum(gt_mask * mask, axis=(1,2))
    union = tf.reduce_sum(gt_mask, axis=(1,2)) + tf.reduce_sum(mask, axis=(1,2)) - intersec
    return 1.0 - intersec / union
