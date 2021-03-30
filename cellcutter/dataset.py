import numpy as np
from numpy.random import default_rng
import tensorflow as tf
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

module_rng = default_rng()

class Dataset:

  def __init__(self, data_img, marker_img, mask_img = None, label_img = None, crop_size = 64, gen_fake_label = True):
    if label_img == None and gen_fake_label:
      label_img = expand_labels(marker_img, distance = 5)

    #if data_img.shape != marker_img.shape or (label_img is not None and data_img.shape != label_img.shape):
    #  raise ValueError('Image size not match each other')

    self.img = self.__normalize_img(data_img)
    self.marker_img = marker_img
    self.label_img = label_img
    if mask_img is not None:
      mask_img = mask_img == 0
      self.mask = np.logical_not(mask_img) * tf.math.log(tf.keras.backend.epsilon())
    else:
      self.mask = None
    self.crop_size = crop_size

    self.create_patches()

  def __normalize_img(self, img_in):
    img = img_in.astype(np.float32)
    if len(img.shape) == 2:
      img = img[..., np.newaxis]
    img -= np.min(img, axis = (0,1))
    img /= np.max(img, axis = (0,1))
    return img

  def __center_of_mass_as_int(self, img):
    d0, d1 = img.shape
    s = np.sum(img)
    c0 = int(np.sum(img, axis = 1).dot(np.arange(d0)) / s + .5)
    c1 = int(np.sum(img, axis = 0).dot(np.arange(d1)) / s + .5)
    return c0,c1

  def __is_within(self, coord, rect):
    return coord[0] >= rect[0] and coord[0] < rect[0] + rect[2] and coord[1] >= rect[1] and coord[1] < rect[1] + rect[3]

  def create_patches(self):
    d0,d1 = self.marker_img.shape
    indices = np.unique(self.marker_img)[1:]  #ignore 0

    self.patch_set = dict()
    _,_,nCh = self.img.shape
    coords = np.zeros((len(indices), 2), dtype = np.int16)
    patches = np.zeros((len(indices), self.crop_size, self.crop_size, nCh+1))
    label_patches = np.zeros((len(indices), self.crop_size, self.crop_size), dtype = np.uint16)

    for i,ind in enumerate(indices):
      c0,c1 = np.round(self.__center_of_mass_as_int(self.marker_img == ind))

      coords[i,:] = [c0,c1]

    coords -= self.crop_size // 2
    coords = np.clip(coords, 0, (d0 - self.crop_size, d1 - self.crop_size))

      #c0 = sorted((0, c0 - self.crop_size // 2, d0 - self.crop_size))[1]
      #c1 = sorted((0, c1 - self.crop_size // 2, d1 - self.crop_size))[1]

      #data_patch_1 = self.img[c0:c0+self.crop_size,c1:c1+self.crop_size,:]
      #data_patch_2 = self.marker_img[c0:c0+self.crop_size,c1:c1+self.crop_size] == ind
      #data_patch = np.concatenate((data_patch_1, data_patch_2[...,np.newaxis]),axis = 2)
    for i,ind in enumerate(indices):
      c0,c1 = coords[i, :]
      patches[i,:,:,:-1] =  self.img[c0:c0+self.crop_size,c1:c1+self.crop_size,:]
      patches[i,:,:,-1] =  self.marker_img[c0:c0+self.crop_size,c1:c1+self.crop_size] == ind

      if self.label_img is not None:
        label_patches[i,:,:] = (self.label_img[c0:c0+self.crop_size,c1:c1+self.crop_size] == ind).astype(np.uint8)
#        self.patch_set[ind] = ((c0,c1), data_patch, label_patch)
#      else:
#        self.patch_set[ind] = ((c0,c1), data_patch)

      if self.label_img is not None:
        self.patch_set = (indices, coords, patches, label_patches)
      else:
        self.patch_set = (indices, coords, patches)

  def generator(self):
    '''
    A generator returning individual patches info: (coord, data)
    '''
    coords = self.patch_set[1]
    patches = self.patch_set[2]

    for i in range(coords.shape[0]):
        yield (coords[i,:], patches[i,...])

  def tf_dataset(self):
    d0,d1,ch = self.img.shape
    return tf.data.Dataset.from_generator(
      self.generator,
      output_signature = (
        tf.TensorSpec(shape=(2,), dtype=tf.int32),
        tf.TensorSpec(shape=(self.crop_size,self.crop_size, ch+1), dtype=tf.float32)
      )
    )

  def generator_with_label(self):
    '''
    A generator returning individual patches pairs with label: (data, label)
    '''
    if self.label_img is None:
      raise ValueError('Lable Image not set.')

    patches = self.patch_set[2]
    labels = self.patch_set[3]
    for i in range(patches.shape[0]):
      yield (patches[i,...], labels[i,...])

  def tf_dataset_with_label(self):
    d0,d1,ch = self.img.shape
    return tf.data.Dataset.from_generator(
      self.generator_with_label,
      output_signature=(
        tf.TensorSpec(shape=(self.crop_size,self.crop_size,ch+1), dtype=tf.float32),
        tf.TensorSpec(shape=(self.crop_size,self.crop_size), dtype=tf.uint8)
      )
    )

  def generator_within_area(self, rng = None, area_size = 640):
    '''
    A generator returning all patches with a certain area, as well as the corresponding coordinates of all pathes
    '''
    if rng is None:
      rng = module_rng

    d0,d1 = self.marker_img.shape
    while True:
      a0 = rng.integers(d0 - area_size - self.crop_size)
      a1 = rng.integers(d1 - area_size - self.crop_size)

      coords = self.patch_set[1]
      patches = self.patch_set[2]
      all_indices_0 = np.logical_and(coords[:,0] >= a0, coords[:,0] < a0 + area_size)
      all_indices_1 = np.logical_and(coords[:,1] >= a1, coords[:,1] < a1 + area_size)
      all_indices = np.logical_and(all_indices_0, all_indices_1)

      coords = coords[all_indices, :] - np.array([a0, a1])
      data = patches[all_indices, ...]

      if self.mask is not None:
        submask = self.mask[a0:a0+self.crop_size+area_size, a1:a1+self.crop_size+area_size]
      else:
        submask = None
      yield data, coords, submask
