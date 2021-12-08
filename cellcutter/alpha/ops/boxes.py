import tensorflow as tf

def box_ious(boxes_a, boxes_b):
    ''' computer ious between pair boxes '''
    area_a = (boxes_a[...,2] - boxes_a[...,0]) * (boxes_a[...,3] - boxes_a[...,1])
    area_b = (boxes_b[...,2] - boxes_b[...,0]) * (boxes_b[...,3] - boxes_b[...,1])
    r0 = tf.maximum(boxes_a[...,0], boxes_b[...,0])
    r1 = tf.minimum(boxes_a[...,2], boxes_b[...,2])
    c0 = tf.maximum(boxes_a[...,1], boxes_b[...,1])
    c1 = tf.minimum(boxes_a[...,3], boxes_b[...,3])
    hh = tf.maximum(r1 - r0, 0.)
    ww = tf.maximum(c1 - c0, 0.)
    ious = hh*ww / (area_a + area_b - hh*ww)
    return tf.cast(ious, tf.float32)

def compare_boxes(boxes, gt_boxes):
    ''' compute relative change between paired boxes'''
    rr = (boxes[...,0] + boxes[...,2])/2
    cc = (boxes[...,1] + boxes[...,3])/2
    hh = boxes[...,2] - boxes[...,0]
    ww = boxes[...,3] - boxes[...,1]

    grr = (gt_boxes[...,0] + gt_boxes[...,2])/2
    gcc = (gt_boxes[...,1] + gt_boxes[...,3])/2
    ghh = gt_boxes[...,2] - gt_boxes[...,0]
    gww = gt_boxes[...,3] - gt_boxes[...,1]

    drr = (grr - rr) / hh
    dcc = (gcc - cc) / ww
    dhh = (ghh - hh) / hh
    dww = (gww - ww) / ww
    return tf.stack([drr, dcc, dhh, dww], axis=-1)

def recover_boxes(boxes, regression_out):
    ''' alter boxes based on relate changes '''
    rr = (boxes[...,0] + boxes[...,2])/2
    cc = (boxes[...,1] + boxes[...,3])/2
    hh = boxes[...,2] - boxes[...,0]
    ww = boxes[...,3] - boxes[...,1]
    rr += regression_out[...,0] * hh
    cc += regression_out[...,1] * ww
    hh += regression_out[...,2] * hh
    ww += regression_out[...,3] * ww
    r0 = rr - hh / 2.
    c0 = cc - ww / 2.
    r1 = rr + hh / 2.
    c1 = cc + ww / 2.
    return tf.stack([r0,c0,r1,c1], axis=-1)

def box_encode(boxes):
    rr = (boxes[...,0] + boxes[...,2])/2
    cc = (boxes[...,1] + boxes[...,3])/2
    hh = boxes[...,2] - boxes[...,0]
    ww = boxes[...,3] - boxes[...,1]
    return tf.stack([rr,cc,hh,ww], axis=-1)

def box_decode(boxes):
    r0 = boxes[...,0] - boxes[...,2]/2
    c0 = boxes[...,1] - boxes[...,3]/2
    r1 = boxes[...,0] + boxes[...,2]/2
    c1 = boxes[...,1] + boxes[...,3]/2
    return tf.stack([r0,c0,r1,c1], axis=-1)
