import numpy as np

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
