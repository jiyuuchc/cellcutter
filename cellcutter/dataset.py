from warnings import warn
from functools import reduce

import numpy as np
from numpy.random import default_rng
import scipy.ndimage

from numpy.lib.stride_tricks import as_strided

module_rng = default_rng()

class Dataset:

  def __init__(self, data_img, marker, mask_img = None, crop_size = 64, sigma = 15):
    '''
    data_img: Cell image. A nd_array of 2 or 3 dimensions. For multi-channel input, the channel is the last axis.
    marker:  Either a nd_array image, or a 2D array (Nx2 or Nx3) listing the location of N markers.
              If an image, the shape should match that of data_img.
    mask_img: Optional nd_array binary mask. 1 is cellular area; 0 is background. Shape should match data_img
    crop_size: The size of each patch.
    '''

    data_img = np.array(data_img, dtype=np.float32)
    marker = np.array(marker)

    if marker.shape == data_img.shape or marker.shape == data_img.shape[:-1]:
      # marker is Image
      img_shape = marker.shape
      rank = len(img_shape)
      marker_img = marker.astype(np.uint)
      indices = np.unique(marker_img)[1:]  #ignore 0
      coords = np.array(scipy.ndimage.measurements.center_of_mass(marker_img, marker_img, indices))
      coords = np.round(coords).astype(int)
    elif len(marker.shape) == 2:
      # markers is a list of positions
      rank = marker.shape[1]
      img_shape = data_img.shape[:rank]
      marker_img = None
      indices = np.arange(marker.shape[0]) + 1
      coords = np.round(marker).astype(int)
    else:
      raise ValueError('Invalid marker input')

    if rank != 2 and rank != 3:
      warn('Input image neither 2D or 3D.')

    crop_size = np.broadcast_to(np.array(crop_size, dtype=int), (rank,))
    sigma = np.broadcast_to(np.array(sigma), (rank,))

    coords -= crop_size // 2
    sel = np.all(np.logical_and(coords >= 0, coords < img_shape - crop_size), 1)
    coords = coords[sel, :]
    indices = indices[sel]

    # last axis of data_img should always to channel
    if len(data_img.shape) <= rank:
      data_img = data_img[..., np.newaxis]
    if len(data_img.shape) != rank + 1:
      raise ValueError('Wrong data_img dimension')

    # Normalize data_img to (0,1)
    nCh = data_img.shape[-1]
    data_img -= np.mean(data_img.reshape(-1, nCh), axis = 0)
    data_img /= np.std(data_img.reshape(-1, nCh), axis = 0)

    if mask_img is not None:
      mask_img = (np.array(mask_img) > 0).astype(np.float32)
      if mask_img.shape != img_shape:
        raise ValueError('Image and mask has different shape.')

    self._crop_size = crop_size
    self._marker_img = marker_img
    self._coordinates = coords
    self._indices = indices
    self._img = data_img
    self._mask = mask_img

    self._pos_embedding = self._get_embedding(crop_size, sigma)

    self._create_patches()

  def _get_embedding(self, patch_size, sigma):
    ll = [np.linspace(-d/2+0.5, d/2-0.5, d)/s for d,s in zip(patch_size, sigma)]
    dst = reduce(lambda a,b: a + b * b / 2, np.meshgrid(*ll), 0)
    img = np.exp(-dst).astype(np.float32)
    return img

  def _create_patches(self):
    indices = self._indices
    coords_t = tuple(self.coordinates.transpose())

    img_shape = self._img.shape[:-1]
    sub_shape = self._crop_size
    view_shape = tuple(np.array(img_shape) - sub_shape + 1) + tuple(sub_shape)

    if self._marker_img is not None:
      marker_view = as_strided(self._marker_img, view_shape, self._marker_img.strides * 2)
      marker_patches = marker_view[coords_t].reshape(len(indices), -1) == indices[:, np.newaxis]
      marker_patches = marker_patches.reshape(len(indices), *sub_shape, 1)
      #marker_patches = marker_view[coords_t] > 0
      #marker_patches = marker_patches[..., np.newaxis]

    nCh = self._img.shape[-1]
    img_view_shape = view_shape + (nCh,)
    img_view_strides = self._img.strides[:-1] + self._img.strides
    img_view = as_strided(self._img, img_view_shape, img_view_strides)
    #self._patches = np.concatenate((img_view[coords_t], marker_patches.astype(img_view.dtype)), axis = -1)

    n_patches = len(self._indices)
    pos_embeddings = np.repeat(self._pos_embedding[np.newaxis, ...], n_patches, axis = 0)

    self._patches = np.concatenate((img_view[coords_t], pos_embeddings[..., np.newaxis]), axis = -1)

  def generator_within_area(self, rng = None, area_size = 640):
    '''
    A generator returning all patches with a certain area, as well as the corresponding coordinates of all pathes
    '''
    if rng is None:
      rng = module_rng

    shape = self._img.shape[:-1]
    rank = len(shape)
    area_size = np.broadcast_to(area_size, (rank,))
    crop_size = self._crop_size

    while True:
      coords = self._coordinates
      c0 = rng.integers(shape - area_size - crop_size)
      all_indices = np.all(coords >= c0, axis = 1)
      all_indices = np.logical_and(all_indices, np.all(coords < c0 + area_size, axis = 1))

      coords = coords[all_indices, :] - c0
      data = self._patches[all_indices, ...]

      if self._mask is not None:
        slices = [slice(*s) for s in zip(c0, c0 + area_size + crop_size)]
        submask = self._mask[tuple(slices)]
      else:
        submask = None

      yield data, coords, submask

  @property
  def marker_img(self):
    return self._marker_img

  @property
  def mask(self):
    return self._mask

  @property
  def patches(self):
    return self._patches

  @property
  def coordinates(self):
    return self._coordinates

  @property
  def indices(self):
    return self._indices

  @property
  def crop_size(self):
    return self._crop_size
