import tensorflow as tf
import numpy as np
from skimage.measure import regionprops

from ..ops.common import *
from ..ops.clustering import *
from ..ops.box_matcher import *

@tf.function(input_signature=(
  tf.TensorSpec(shape=(None,None), dtype=tf.int32),
  tf.TensorSpec(shape=(None,4), dtype=tf.float32),
  tf.TensorSpec(shape=(), dtype=tf.int32),
))
def crop_pred(preds, bbox, crop_size=64):
    n_proposals = tf.reduce_max(preds) + 1
    pred_imgs = tf.map_fn(
        lambda k: tf.cast(preds == k, tf.int32),
        tf.range(n_proposals),
        fn_output_signature = tf.TensorSpec(shape=(None,None), dtype=tf.int32),
    )

    pred_imgs = tf.expand_dims(pred_imgs, -1)

    return tf.image.crop_and_resize(
        tf.cast(pred_imgs, tf.float32),
        bbox,
        tf.range(tf.shape(bbox)[0]),
        crop_size=(crop_size,crop_size),
    )

@tf.function(input_signature=(
  tf.TensorSpec(shape=(None,None,1), dtype=tf.float32),
  tf.TensorSpec(shape=(None,4), dtype=tf.float32),
  tf.TensorSpec(shape=(), dtype=tf.int32),
))
def crop_img(img, bbox, crop_size=64):
    img = img[None,...]
    return tf.image.crop_and_resize(
        img,
        bbox,
        tf.zeros(shape=(tf.shape(bbox)[0],), dtype=tf.int32),
        crop_size=(crop_size,crop_size),
    )

@tf.function(input_signature=(
  tf.TensorSpec(shape=(None,3), dtype=tf.int32),
  tf.TensorSpec(shape=(None,4), dtype=tf.float32),
  tf.TensorSpec(shape=(None,), dtype=tf.int32),
  tf.TensorSpec(shape=(), dtype=tf.int32),
  tf.TensorSpec(shape=(), dtype=tf.int32),
  tf.TensorSpec(shape=(), dtype=tf.int32),
))
def crop_masks(mask_indices, bbox, matched, crop_size=64, h=544, w=704):
    n_masks = mask_indices[-1,0] + 1
    masks = tf.scatter_nd(mask_indices, tf.ones(shape=(tf.shape(mask_indices)[0],), dtype=tf.float32), shape=(n_masks, h, w))
    masks = tf.expand_dims(masks,-1)
    return tf.image.crop_and_resize(
        masks,
        bbox,
        matched,
        crop_size=(crop_size, crop_size),
    )

@tf.function(input_signature=(
  tf.TensorSpec(shape=(None,4), dtype=tf.int32),
  tf.TensorSpec(shape=(), dtype=tf.int32),
  tf.TensorSpec(shape=(), dtype=tf.int32),
  tf.TensorSpec(shape=(), dtype=tf.int32),
))
def adjust_bbox(bbox, min_size=64, h=544, w=704):
    '''
    adjust bbox size to be 2x larger, but no smaller than min_size x min_size
    '''
    cy = (bbox[:,0] + bbox[:,2])/2
    cx = (bbox[:,1] + bbox[:,3])/2
    hh = (bbox[:,2] - bbox[:,0])*2
    ww = (bbox[:,3] - bbox[:,1])*2
    hh = tf.clip_by_value(hh, min_size, 9999)
    ww = tf.clip_by_value(ww, min_size, 9999)
    h = tf.cast(h, tf.float64)
    w = tf.cast(w, tf.float64)
    bbox_out = tf.stack([
          (cy - hh / 2) / h,
          (cx - ww / 2) / w,
          (cy + hh / 2) / h,
          (cx + ww / 2) / w,
      ], axis = -1)
    return tf.cast(bbox_out, tf.float32)

def bbox_of_preds(preds_one_img): # for one image
    h,w = preds_one_img.shape
    bbox = np.array([r.bbox for r in regionprops(preds_one_img + 1)])
    return bbox.astype(np.int32)

def parser(inputs, det_model, min_iou = 0.1):
    matcher=BoxMatcher([min_iou,], [0,1])
    img, labels = inputs

    model_out=det_model((img[None,...], tf.nest.map_structure(lambda x: x[None,...], labels)))
    preds = pred_labels(model_out['offsets'], model_out['weights'])
    preds = preds[0]  # data is not batched

    similarity = proposal_iou(preds, labels['mask_indices'])
    matched, ind = matcher(similarity)

    bbox = tf.numpy_function(bbox_of_preds, [preds], tf.int32)
    bbox = adjust_bbox(bbox)
    pred_crops = crop_pred(preds, bbox)
    img_crops = crop_img(img, bbox)
    mask_crops = crop_masks(labels['mask_indices'], bbox, matched)
    ind = ind * (labels['class'] + 1)

    return {
          'source_image': img_crops,
          'proposal': pred_crops,
          'gt_cell_type': ind,
          'gt_mask': mask_crops,
          'bbox': bbox,
          }

# output_signature = {
#         'prediction': tf.TensorSpec(shape=(None,64,64,1), dtype=tf.float32),
#         'source_image': tf.TensorSpec(shape=(None,64,64,1), dtype=tf.float32),
#         'gt_cell_type': tf.TensorSpec(shape=(None,), dtype=tf.int32),
#         'gt_mask': tf.TensorSpec(shape=(None,64,64,1), dtype=tf.float32),
#     }
