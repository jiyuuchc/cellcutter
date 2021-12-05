'''simple parser for the DetModel'''
import tensorflow as tf
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from skimage.filters import threshold_otsu

def _compute_weights(mask, center):
    center = np.round(center).astype(int)
    dist_to_edge = distance_transform_edt(mask)
    img = np.ones_like(mask)
    img[center[0], center[1]] = 0
    dist_to_center = distance_transform_edt(img)
    return dist_to_edge / (dist_to_edge + dist_to_center + 1e-16)

def _fix_label(masks, bboxes):
    def _clean_mask(mask, th=40):
        # mask = mask > threshold_otsu(np.array(mask).astype(np.uint8))
        mask = binary_fill_holes(mask).astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        c = contours[0][:, 0]
        diff = c - np.roll(c, 1, 0)
        targets = (diff[:, 1] == 0) & (np.abs(diff[:, 0]) >= th)  # find horizontal lines longer than threshold
        return mask, (True in targets)
    broken_flags = []
    for bbox, mask in zip(bboxes, masks):
        r0,c0,r1,c1 = bbox
        m = mask[r0:r1, c0:c1]
        m2, is_broken = _clean_mask(m)
        mask[r0:r1, c0:c1] = m2
        broken_flags.append(is_broken)
    return masks, np.array(broken_flags)

def _gen_training_data(masks, bboxes):
    _, height, width = masks.shape
    weights = []
    all_locs = []
    for bbox, mask in zip(bboxes, masks):
        r0,c0,r1,c1 = bbox
        m = mask[r0:r1, c0:c1]
        loc = np.array(np.where(m)).mean(axis=-1)
        all_locs.append(loc + [r0, c0])

        w = _compute_weights(m, loc)
        tmp = np.zeros((height, width), np.float32)
        tmp[r0:r1, c0:c1] = w
        weights.append(tmp)

    weights = np.array(weights)
    indicator = np.expand_dims(np.argmax(weights, axis = 0), 0)
    weights = np.take_along_axis(weights, indicator, 0).squeeze()
    weights = np.expand_dims(weights, -1)

    locs = np.array(all_locs)
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    dx = x[None, :, :] - locs[:,None,None,1]
    dy = y[None, :, :] - locs[:,None,None,0]
    dx = np.take_along_axis(dx, indicator, 0).squeeze()
    dy = np.take_along_axis(dy, indicator, 0).squeeze()
    dist_map = np.stack([dy,dx], -1) * np.tile(tf.cast(weights>0, tf.float32), [1,1,2])

    heights = np.take_along_axis(
        np.tile(np.expand_dims(bboxes[:,2]-bboxes[:,0],axis=(1,2)), [1, height, width]),
        indicator, 0,
    ).squeeze()
    widths = np.take_along_axis(
        np.tile(np.expand_dims(bboxes[:,3]-bboxes[:,1],axis=(1,2)), [1, height, width]),
        indicator, 0,
    ).squeeze()
    size_map = np.stack([heights, widths], -1) * np.tile(tf.cast(weights>0, tf.float32), [1,1,2])

    return weights, dist_map.astype(np.float32), size_map.astype(np.float32)

def parser(image, labels, h=544, w=704):
    height = labels['height']
    width = labels['width']

    mi = labels['mask_indices']
    n_masks = mi[-1,0] + 1
    masks = tf.scatter_nd(
        mi,
        tf.ones([tf.shape(mi)[0]], tf.uint8),
        [n_masks, height, width]
    )
    bboxes = labels['bboxes']
    masks, is_broken = tf.numpy_function(
        _fix_label,
        [masks, bboxes],
        [tf.uint8, tf.bool]
    )
    weights,dist_map, size_map = tf.numpy_function(
        _gen_training_data,
        [masks, bboxes],
        [tf.float32, tf.float32, tf.float32],
    )

    dist_map = tf.image.resize_with_crop_or_pad(dist_map, h, w)
    size_map = tf.image.resize_with_crop_or_pad(size_map, h, w)
    weights = tf.image.resize_with_crop_or_pad(weights, h, w)
    image = tf.image.resize_with_crop_or_pad(image, h, w)

    mask_indices = tf.cast(tf.where(masks), tf.int32)
    mask_indices = mask_indices + [0, (h - height)//2, (w - width)//2]
    bboxes = bboxes + [(h - height)//2, (w - width)//2, (h - height)//2, (w - width)//2]

    dist_map = tf.ensure_shape(dist_map, (h, w, 2))
    size_map = tf.ensure_shape(size_map, (h, w, 2))
    weights = tf.ensure_shape(weights, (h, w, 1))
    is_broken = tf.ensure_shape(is_broken, (None,))

    new_labels = {
        'source_id': labels['source_id'],
        'height': labels['height'],
        'width': labels['width'],
        'class': labels['class'],
        'bboxes': bboxes,
        'mask_indices': mask_indices,
        'dist_map': dist_map,
        'size_map': size_map,
        'weights': weights,
        'is_broken': is_broken,
    }
    return image, new_labels

def augment(image, labels):
    image = image[...,::-1,:]
    width = tf.shape(image)[-2]
    mask_indices = labels['mask_indices']
    dist_map = labels['dist_map'][:,::-1,:] * [1,-1]
    size_map = labels['size_map'][:,::-1,:]
    weights = labels['weights'][:,::-1,:]
    mi = labels['mask_indices']*[1,1,-1] + [0,0,width-1]
    bboxes = labels['bboxes'] *[1,-1,1,-1] + [0,width-1,0,width-1]
    bboxes = tf.gather(bboxes, (0,3,2,1), axis=-1)

    new_labels = {}
    new_labels.update(labels)
    # for k in labels:
    #     new_labels[k] = labels[k]
    new_labels['dist_map'] = dist_map
    new_labels['size_map'] = size_map
    new_labels['weights'] = weights
    new_labels['bboxes'] = bboxes
    new_labels['mask_indices'] = mi

    return image, new_labels
