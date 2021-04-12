import numpy as np

from skimage.feature import blob_doh
from skimage.measure import regionprops
from sklearn.preprocessing import StandardScaler
from skimage.filters import gaussian

from .train import expand_labels

def label_with_blob_detection(img, max_sigma = 10, min_sigma = 3, threshold = .01):
  ''' Generate maker label from nucleus imag using blob detection
  '''
  img = np.array(img, dtype = np.double)
  img = StandardScaler().fit_transform(img.reshape(-1,1)).reshape(img.shape)
  blobs = blob_doh(img, max_sigma = max_sigma, min_sigma = min_sigma, threshold = threshold)

  # remove dark on bright blobs
  xs, ys = np.round(blobs[:,:2]).astype(int).transpose()
  blobs_p = blobs[np.greater(gaussian(img)[(xs, ys)],0), :]
  xs, ys = np.round(blobs_p[:,:2]).astype(int).transpose()

  label = np.zeros(shape = img.shape, dtype=int)
  label[(xs,ys)] = np.arange(len(xs)) + 1

  label = expand_labels(label, max_sigma - 2)
  return label
