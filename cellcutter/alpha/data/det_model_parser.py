'''simple parser for the DetModel'''

def parser(image, labels):
    dist_map = labels['dist_map']
    weights = labels['weights']
    mask_indices = labels['mask_indices']
    flipped = tf.random.uniform(()) >= 0.5
    if flipped:
        dist_map = dist_map[...,::-1,:] * [1,-1]
        weights = weights[...,::-1,:]
        image = image[...,::-1,:]
        mask_indices = mask_indices * [1,1,-1] + [0,0,labels['width']-1]
    dist_map = tf.image.resize_with_crop_or_pad(dist_map, 640,768)
    weights = tf.image.resize_with_crop_or_pad(weights, 640,768)
    image = tf.image.resize_with_crop_or_pad(image, 640,768)
    mask_indices = mask_indices + [0, (640 - labels['height'])//2, (768-labels['width'])//2]
    labels.update({
        'dist_map': dist_map,
        'weights': weights,
        'flipped': flipped,
        'mask_indices': mask_indices,
    })
    return image, labels
