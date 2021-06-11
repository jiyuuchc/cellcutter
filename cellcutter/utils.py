import numpy as np
import tensorflow as tf
from skimage.morphology import binary_erosion
from skimage.morphology import disk
from skimage import img_as_ubyte
from skimage.filters.rank import entropy
import sklearn.mixture
from warnings import warn

from numpy.lib.stride_tricks import as_strided

try:
  import maxflow
except ModuleNotFoundError:
  warn("PyMaxFlow not found. Some functions may not work.")

def draw_label(data, model, image):
  '''
  data: cellcutter.dataset object
  model: a tf NN model
  image: a 2D array to be drawn on
  '''
  coords = data.coordinates
  preds = tf.sigmoid(model(data.patches)).numpy().squeeze()

  d0,d1 = preds.shape[-2:]
  max_prob = np.zeros(image.shape, dtype=np.float32)

  # find the max prob at each pixel
  # we have to go through patchs sequentially to have a predictable execution order
  for coord, pred in zip(coords, preds):
    c0,c1 = list(coord)
    max_prob[c0:c0+d0,c1:c1+d1] = np.maximum(max_prob[c0:c0+d0,c1:c1+d1], pred)

  label = 1
  # now remove any pred output that is not the max
  for coord, pred in zip(coords, preds):
    c0,c1 = list(coord)
    pred[pred < max_prob[c0:c0+d0,c1:c1+d1]] = 0
    image[c0:c0+d0, c1:c1+d1] = np.maximum(image[c0:c0+d0, c1:c1+d1], (pred > 0.5) * label)
    label += 1

  return image

def draw_border(data, model, image, batch_size = 256):
  '''
  data: cellcutter.dataset object
  model: a tf NN model
  image: a 2D array to be drawn on
  '''
  coords = data.coordinates
  preds = tf.sigmoid(model(data.patches)).numpy().squeeze()
  d0,d1 = preds.shape[-2:]

  # add a border so that erosion operation always work
  preds = np.pad(preds, ((0,0),(1,1),(1,1)))

  for coord, pred in zip(coords, preds):
    c0,c1 = list(coord)
    patch = (pred > 0.5).astype(np.uint8)
    edge = patch - binary_erosion(patch)
    edge = edge[1:-1,1:-1] # remove extra border added above
    image[c0:c0+d0,c1:c1+d1] += edge

  return image

def img2porb(img):
  '''
  Cluster image values using a 2-state (Cell/Background) Gaussian Mixture model.
  Return probabilities of states for each pixel
  '''
  gmm_model = sklearn.mixture.GaussianMixture(2)
  gmm_model.fit(img.flatten()[...,np.newaxis])
  prob = gmm_model.predict_proba(entr_img.flatten()[...,np.newaxis])[:,0].reshape(entr_img.shape)
  return prob

def graph_cut(data_img, prior = 0.5, max_weight = 5, sigma = 0.1):
  ''' A generic graph cut algorithm to separate foreground from background.
  data_img: either 2D or 3D numpy array
  prior: Prior probability of foreground pixels, float (0,1)
  max_weight: Max cut penalty. A smaller value resulted in more fragmented cutting
  sigma: Affects how much the intensity gradient lowers the cut penalty. A large value means low effect and thus a more constant cut penalty.

  returns: Graph cut image
  '''
  if prior <=0 or prior >= 1.0:
    raise ValueError("prior must be between (0,1)")

  dim = len(data_img.shape)
  if dim != 2 and dim !=3:
    raise ValueError("Input image should be 2D or 3D")

  img = (data_img - data_img.min()) / data_img.ptp()

  f_weights = -np.log(np.maximum(img, np.finfo(float).eps)) - np.log(prior)
  b_weights = -np.log(1.0-prior) - np.log(np.maximum(1.0 - img, np.finfo(float).eps))

  g = maxflow.GraphFloat()
  nodes = g.add_grid_nodes(img.shape)

  g.add_grid_tedges(nodes, f_weights, b_weights)

  if dim == 2:
    connectivities = ((-1,1), (0,1), (1,1), (1,0)) # 2D
    for c in connectivities:
      struct = np.zeros((3,3))
      struct[1+c[0], 1+c[1]] = 1
      weights = img - np.roll(img, -np.array(c), axis=(0,1))
      weights = np.exp(-weights*weights/2/sigma/sigma) * max_weight
      g.add_grid_edges(nodes, weights, struct)
  else:
    connectivities = ((0,-1,1), (0,0,1), (0,1,0), (0,1,1), (1,-1,-1), (1,-1,0), (1,-1,1), (1,0,-1), (1,0,0), (1,0,1), (1,1,-1), (1,1,0), (1,1,1)) # 3D
    for c in connectivities:
      struct = np.zeros((3,3,3))
      struct[1+c[0], 1+c[1], 1+c[2]] = 1
      weights = img - np.roll(img, -np.array(c), axis=(0,1,2))
      weights = np.exp(-weights*weights/2/sigma/sigma) * max_weight
      g.add_grid_edges(nodes, weights, struct)

  g.maxflow()
  sgm_img = g.get_grid_segments(nodes)

  return sgm_img
