import numpy as np
import tensorflow as tf
from skimage.morphology import binary_erosion
from skimage.morphology import disk
from skimage import img_as_ubyte
from skimage.filters.rank import entropy
import sklearn.mixture
import maxflow
try:
  from skimage.segmentation import expand_labels
except ImportError:
  '''
  The expand_labels() is not implemented in earlier versions of skimage
  So it is directy copied here if import fails
  '''
  from scipy.ndimage import distance_transform_edt
  def expand_labels(label_image, distance=1):
      distances, nearest_label_coords = distance_transform_edt(
          label_image == 0, return_indices=True
      )
      labels_out = np.zeros_like(label_image)
      dilate_mask = distances <= distance
      masked_nearest_label_coords = [
          dimension_indices[dilate_mask]
          for dimension_indices in nearest_label_coords
      ]
      nearest_labels = label_image[tuple(masked_nearest_label_coords)]
      labels_out[dilate_mask] = nearest_labels
      return labels_out

def draw_label(data, model, image, batch_size = 256):
  '''
  data: cellcutter.dataset object
  model: a tf NN model
  image: a 2D array to be drawn on
  '''
  label = 1
  dataset = data.tf_dataset()
  for coords, patches, *_ in dataset.batch(batch_size):
    coords = tf.unstack(coords)
    preds = tf.unstack(tf.squeeze(tf.math.sigmoid(model(patches))))
    for coord, pred in zip(coords, preds):
      c0,c1 = list(coord)
      patch = (pred.numpy() >= 0.5)* label
      d0,d1 = patch.shape
      image[c0:c0+d0,c1:c1+d1] = np.maximum(image[c0:c0+d0,c1:c1+d1], patch)
      label += 1
  return image

def draw_border(data, model, image, batch_size = 256):
  '''
  data: cellcutter.dataset object
  model: a tf NN model
  image: a 2D array to be drawn on
  '''
  dataset = data.tf_dataset()
  for coords, patches, *_ in dataset.batch(batch_size):
    coords_stack = tf.unstack(coords)
    preds = tf.squeeze(tf.math.sigmoid(model(patches)))
    preds_pad = tf.pad(preds, ((0,0),(1,1),(1,1)))
    preds_stack = tf.unstack(preds_pad)
    for coord, pred in zip(coords_stack, preds_stack):
      c0,c1 = list(coord)
      patch = (pred.numpy() >= 0.5).astype(np.uint8)
      edge = patch - binary_erosion(patch)
      edge = edge[1:-1,1:-1]
      d0,d1 = edge.shape
      image[c0:c0+d0,c1:c1+d1] += edge
  return image

def gen_mask_from_data(data_img, entropy_disk_size = 16, graph_cut_weight = 5):
  ''' generate a mask for the cell area from the data Image
  Step1: perform local entropy calcualation. The area with cells are expected to have higer entropy_disk_size
  Step2: Cluster entropy values using a 2-state (Cell/Background) Gaussian Mixture model and compute probabilities of states for each pixel.
  Step3: Perform a maxflow graph cut to separate cell and background region

  returns: (mask_image, entropy_image)
  '''
  img_u = img_as_ubyte((data_img - data_img.min()) / data_img.ptp())
  entr_img = entropy(img_u, disk(entropy_disk_size))
  entr_img = (entr_img - entr_img.min()) / entr_img.ptp()

  gmm_model = sklearn.mixture.GaussianMixture(2)
  gmm_model.fit(entr_img.flatten()[...,np.newaxis])
  prob = gmm_model.predict_proba(entr_img.flatten()[...,np.newaxis])[:,0].reshape(entr_img.shape)

  g = maxflow.GraphFloat()
  nodes = g.add_grid_nodes(entr_img.shape)
  g.add_grid_edges(nodes, graph_cut_weight)
  g.add_grid_tedges(nodes, np.log(1-prob), np.log(prob))
  g.maxflow()
  sgm_img = g.get_grid_segments(nodes)

  return sgm_img, entr_img

def gen_fake_labels(marker_label, dist = 5):
  '''generate fake segmentation labels by simply expand the marker labels
  '''
  return expand_labels(marker_label, distance = dist)
