''' Data generator for Live-cell dataset from Sartorious'''

import os
import tensorflow as tf
import numpy as np
import pandas as pd
import imageio
import cv2
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from skimage.filters import threshold_otsu

def annotation_to_indices(annotation, dense_shape):
    '''
    annotation: string
    dense_shape: (height,width) of original image
    Returns: indices
    '''
    annotation = np.array(annotation.split(), dtype=int).reshape(-1,2)
    indices = np.concatenate([np.arange(s,s+l) for s,l in annotation])-1
    indices = np.unravel_index(indices, dense_shape)
    return np.array(indices)

def indices_to_mask(ind):
    '''
    ind: array
    Returns: (R,C,M) topleft corner and mask image as np array
    '''
    R0,C0 = ind.min(axis = 1)
    R1,C1 = ind.max(axis = 1) + 1
    ind_ = ind - np.array([R0,C0])[:,None]
    img = np.zeros([R1-R0, C1-C0], dtype = int)
    img[tuple(ind_)] = 1
    return R0,C0,img

def compute_weights(mask, center):
    center = np.round(center).astype(int)
    dist_to_edge = distance_transform_edt(mask)
    img = np.ones_like(mask)
    img[center[0], center[1]] = 0
    dist_to_center = distance_transform_edt(img)
    return dist_to_edge / (dist_to_edge + dist_to_center + 1e-16)

def clean_mask(mask, th=40):
    mask = mask > threshold_otsu(np.array(mask).astype(np.uint8))
    mask = binary_fill_holes(mask).astype(np.uint8)
    # New code for mask acceptance
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = contours[0][:, 0]
    diff = c - np.roll(c, 1, 0)
    targets = (diff[:, 1] == 0) & (np.abs(diff[:, 0]) >= th)  # find horizontal lines longer than threshold
    return mask, (True in targets)

def generator(df, img_path):
    cell_types = {'shsy5y': 0, 'astro': 1, 'cort': 2}
    for img_id in df.index.unique():
        img = imageio.imread(os.path.join(img_path, img_id + '.png')) / 255

        # weights=[]
        # locs=[]
        all_indices = []
        all_bboxes = []
        for k, ann in enumerate(df.loc[img_id, 'annotation']):
            indices = annotation_to_indices(ann, img.shape)
            R0,C0 = indices.min(axis = 1)
            R1,C1 = indices.max(axis = 1) + 1
            bbox = [R0,C0,R1,C1]
            indices = np.insert(indices,0,k,axis=0).transpose()
            all_indices.append(indices)
            all_bboxes.append(bbox)
            # r,c,mask = indices_to_mask(indices)
            # loc = indices.mean(axis=1)
            # w = compute_weights(mask, loc - [r,c])
            # locs.append(loc)
            # tmp = np.zeros(shape=img.shape, dtype=np.float32)
            # tmp[r:r+w.shape[0], c:c+w.shape[1]] = w
            # weights.append(tmp)
        # locs = np.array(locs)
        # weights = np.array(weights)
        # indicator = np.expand_dims(np.argmax(weights, axis = 0), 0)
        all_indices = np.concatenate(all_indices, axis=0)
        all_bboxes = np.array(all_bboxes)

        # weights = np.take_along_axis(weights, indicator, 0).squeeze()
        # weights = tf.expand_dims(weights, -1)

        # x, y = np.meshgrid(np.arange(img.shape[-1]), np.arange(img.shape[-2]))
        # dx = x[None, :, :] - locs[:,None,None,1]
        # dy = y[None, :, :] - locs[:,None,None,0]
        # dx = np.take_along_axis(dx, indicator, 0).squeeze()
        # dy = np.take_along_axis(dy, indicator, 0).squeeze()
        # dist_map = np.stack([dy,dx], -1) * tf.tile(tf.cast(weights>0, tf.float32), [1,1,2])

        cls = cell_types[df.loc[img_id,'cell_type'][0]]

        labels = {
            'source_id': img_id,
            'height': img.shape[0],
            'width': img.shape[1],
            'class': cls,
#            'dist_map': dist_map,
#            'weights': weights,
            'mask_indices': all_indices,
            'bboxes': all_bboxes,
        }

        yield img[:,:,None], labels

def get_output_signature(h=520, w=704):
    return (
      tf.TensorSpec(shape=(h, w, 1), dtype=tf.float32),
      {
          'source_id': tf.TensorSpec(shape=(), dtype=tf.string),
          'height': tf.TensorSpec(shape=(), dtype=tf.int32),
          'width': tf.TensorSpec(shape=(), dtype=tf.int32),
          'class': tf.TensorSpec(shape=(), dtype=tf.int32),
          # 'dist_map': tf.TensorSpec(shape=(h,w,2), dtype=tf.float32),
          # 'weights': tf.TensorSpec(shape=(h,w,1), dtype=tf.float32),
          'mask_indices': tf.TensorSpec(shape=(None,3), dtype=tf.int32),
          'bboxes': tf.TensorSpec(shape=(None,4), dtype=tf.int32)
      },
    )

def as_dataset(dataframe, img_dir, img_h = 520, img_w = 704):
    return tf.data.Dataset.from_generator(
        lambda: generator(dataframe, img_dir),
        output_signature = get_output_signature(img_h, img_w),
        )
