import numpy as np
from scipy import ndimage
from numpy.random import default_rng

rng = default_rng()

def normalize_img(img_in):
  img = img_in.astype(np.float32)
  if len(img.shape) == 2:
    img = img[..., np.newaxis]
  d0,d1,ch = img.shape
  img -= np.min(img, axis = (0,1))
  img /= np.max(img, axis = (0,1))
  return img

def get_coords(marker_img, crop_size):
  d0,d1 = marker_img.shape
  indices = np.unique(marker_img)[1:]  #ignore 0

  coords = np.zeros((indices.size, 2), dtype=np.int) # coordinates of each cropped image

  for i, ind in enumerate(indices):
    coords[i,...] = np.round(ndimage.measurements.center_of_mass(marker_img == ind))
  coords -= crop_size // 2
  coords[:,0] = np.clip(coords[:,0], 0, d0 - crop_size)
  coords[:,1] = np.clip(coords[:,1], 0, d1 - crop_size)

  return coords, indices

def is_within(coord, rect):
  return coord[0] >= rect[0] and coord[0] < rect[0] + rect[2] and coord[1] >= rect[1] and coord[1] < rect[1] + rect[3]

def gen_data_model_b(data_img, marker_img, crop_size = 64, area_size = 320):
  '''
  img: (d0,d1,ch) image data
  marker_img: (d0,d1), indexed binary image labeling each marker (e.g. nucleus)
  crop_size: size ofeach cropped area
  area_size: Size of area the crops are from
  '''
  coords, indices = get_coords(marker_img, crop_size)
  img = normalize_img(data_img)

  d0,d1,ch = img.shape
  img_stack = np.zeros((len(indices), crop_size, crop_size, ch + 1), dtype = np.float32)
  for i, ind in enumerate(indices):
    c0 = int(coords[i,0])
    c1 = int(coords[i,1])
    img_stack[i,:,:,0:ch] = img[c0:c0+crop_size, c1:c1+crop_size, :]
    img_stack[i,:,:,-1] = marker_img[c0:c0+crop_size, c1:c1+crop_size] == ind

  while True:
    a0 = rng.integers(d0 - area_size - crop_size)
    a1 = rng.integers(d1 - area_size - crop_size)

    all_coord_indices = [ i for i in range(len(indices)) if is_within(coords[i,:], (a0, a1, area_size, area_size)) ]
    yield (img_stack[all_coord_indices, ...], coords[all_coord_indices,:])

def gen_data_model_a(img, marker_img, seg_img, out_size=64):
  '''
  img: (d0,d1,ch) image data
  marker_img: (d0,d1), indexed binary image labeling each marker (e.g. nucleus)
  seg_img: (d0,d1), indexed segmenentaion image labeling each cell. Ground truth
  '''

  img = normalize_img(img)

  max_index = np.max(marker_img)

  x = np.zeros((out_size, out_size, ch + 1), dtype = np.float32)
  y = np.zeros((out_size, out_size), dtype = np.uint8)

  for i in range(max_index):
    m1 = marker_img == i + 1
    m2 = seg_img == i + 1

    if ( not np.any(m1) or not np.any(m2)):
      continue

    c0,c1 = ndimage.measurements.center_of_mass(m1)
    c0 = int(np.clip(np.round(c0 - out_size // 2), 0, d0 - out_size))
    c1 = int(np.clip(np.round(c1 - out_size // 2), 0, d1 - out_size))

    x[:,:,0:ch] = img[c0:c0+out_size, c1:c1+out_size, :]
    x[:,:,-1] = m1[c0:c0+out_size, c1:c1+out_size] * 1.0
    y = m2[c0:c0+out_size, c1:c1+out_size]

    yield (x, y)
