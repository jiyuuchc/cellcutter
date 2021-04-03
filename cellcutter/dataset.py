import numpy as np
from numpy.random import default_rng
import tensorflow as tf
import scipy.ndimage

from numpy.lib.stride_tricks import as_strided

module_rng = default_rng()

class Dataset:

  def __init__(self, data_img, marker_img, mask_img = None, label_img = None, crop_size = 64):
    #if label_img == None and gen_fake_label:
    #  label_img = expand_labels(marker_img, distance = 5)

    #if data_img.shape != marker_img.shape or (label_img is not None and data_img.shape != label_img.shape):
    #  raise ValueError('Image size not match each other')

    self.rank = len(marker_img.shape)
    self.img = self.__normalize_img(np.array(data_img, dtype=np.float32))
    self.marker_img = np.array(marker_img, dtype=np.uint)
    self.label_img = np.array(label_img, dtype=np.uint)
    if mask_img is not None:
      mask_img = np.array(mask_img)
      self.mask = ((mask_img > 0) * np.log(tf.keras.backend.epsilon())).astype(np.float32)
    else:
      self.mask = None
    self.crop_size = np.array(crop_size, dtype=int)

    if self.mask is not None:
      if self.rank != len(self.mask.shape) :
        raise ValueError('marker and mask has different rank.')

    self.create_patches()

  def __normalize_img(self, img):
    if len(img.shape) < self.rank + 1:
      img = img[..., np.newaxis]
    if len(img.shape) != self.rank + 1:
      raise ValueError(
        'Inut image has wrong dimension. It should either be %i (single channel) or %i (multichannel)' % (self.rank, self.rank+1)
        )
    nCh = img.shape[-1]
    img -= np.min(img.reshape(-1, nCh), axis = 0)
    img /= np.max(img.reshape(-1, nCh), axis = 0)
    return img

  def create_patches(self):
    #d0,d1 = self.marker_img.shape
    indices = np.unique(self.marker_img)[1:]  #ignore 0
    self.__indices = indices

    coords = np.array(scipy.ndimage.measurements.center_of_mass(self.marker_img, self.marker_img, indices)) + 0.5
    coords = coords.astype(np.int) - self.crop_size // 2
    coords = np.clip(coords, 0, self.marker_img.shape - self.crop_size)
    self.__coordinates = coords

    coords_t = tuple(coords.transpose())

    sub_shape = np.broadcast_to(self.crop_size, (self.rank,))
    view_shape = tuple(self.marker_img.shape - sub_shape + 1) + tuple(sub_shape)
    if self.label_img is not None:
      label_view = as_strided(self.label_img, view_shape, self.label_img.strides * 2)
      labels = label_view[coords_t].reshape(len(indices), -1) == indices[:, np.newaxis]
      self.__patch_labels = labels.reshape(len(indices), *sub_shape)

    marker_view = as_strided(self.marker_img, view_shape, self.marker_img.strides * 2)
    marker_patches = marker_view[coords_t].reshape(len(indices), -1) == indices[:, np.newaxis]
    marker_patches = marker_patches.reshape(len(indices), *sub_shape, 1)

    nCh = self.img.shape[-1]
    img_view_shape = view_shape + (nCh,)
    img_view_strides = self.img.strides[:-1] + self.img.strides
    img_view = as_strided(self.img, img_view_shape, img_view_strides)
    self.__patches = np.concatenate((img_view[coords_t], marker_patches.astype(img_view.dtype)), axis = -1)

  def tf_dataset_with_label(self):
    imgs = tf.data.Dataset.from_tensor_slices(self.__patches)
    labels = tf.data.Dataset.from_tensor_slices(self.__patch_labels)
    return tf.data.Dataset.zip((imgs, labels))

  def tf_dataset(self):
    coords = tf.data.Dataset.from_tensor_slices(self.__coordinates)
    imgs = tf.data.Dataset.from_tensor_slices(self.__patches)
    return tf.data.Dataset.zip((coords, imgs))

  def generator_within_area(self, rng = None, area_size = 640):
    '''
    A generator returning all patches with a certain area, as well as the corresponding coordinates of all pathes
    '''
    if rng is None:
      rng = module_rng

    shape = self.img.shape[:-1]
    area_size = np.broadcast_to(area_size, (self.rank,))
    crop_size = np.broadcast_to(self.crop_size, (self.rank,))

    while True:
      coords = self.__coordinates
      c0 = rng.integers(shape - area_size - crop_size)
      all_indices = np.all(coords >= c0, axis = 1)
      all_indices = np.logical_and(all_indices, np.all(coords < c0 + area_size, axis = 1))

      coords = coords[all_indices, :] - c0
      data = self.__patches[all_indices, ...]

      if self.mask is not None:
        slices = [slice(*s) for s in zip(c0, c0 + area_size + crop_size)]
        submask = self.mask[tuple(slices)]
      else:
        submask = None

      yield data, coords, submask
