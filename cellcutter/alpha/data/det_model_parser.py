'''simple parser for the DetModel'''
import tensorflow as tf

def parser(image, labels, random_horizontal_flip = True, h=544, w=704):
    dist_map = labels['dist_map']
    weights = labels['weights']
    mask_indices = labels['mask_indices']
    flipped = False
    if random_horizontal_flip and tf.random.uniform(()) >= 0.5:
        flipped = True
        dist_map = dist_map[...,::-1,:] * [1,-1]
        weights = weights[...,::-1,:]
        image = image[...,::-1,:]
        mask_indices = mask_indices * [1,1,-1] + [0,0,labels['width']-1]
    dist_map = tf.image.resize_with_crop_or_pad(dist_map, h, w)
    weights = tf.image.resize_with_crop_or_pad(weights, h, w)
    image = tf.image.resize_with_crop_or_pad(image, h, w)
    mask_indices = mask_indices + [0, (h - labels['height'])//2, (w-labels['width'])//2]
    labels.update({
        'dist_map': dist_map,
        'weights': weights,
        'flipped': flipped,
        'mask_indices': mask_indices,
    })
    return image, labels
