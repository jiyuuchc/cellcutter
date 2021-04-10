import tensorflow as tf
import numpy as np

from numpy.random import default_rng
from numpy.lib.stride_tricks import as_strided

from .loss import cutter_loss

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

def augment(img, t):
  if t & 1:
    img = tf.image.flip_left_right(img)
  if t & 2:
    img = tf.image.flip_up_down(img)
  if t & 4:
    img = tf.image.transpose(img)
  return img

def augment_r(img, t):
  if t & 4:
    img = tf.image.transpose(img)
  if t & 1:
    img = tf.image.flip_left_right(img)
  if t & 2:
    img = tf.image.flip_up_down(img)
  return img

def _gen_fake_label(data, expand_size = 5):
  marker = data.marker_img
  fake_label = expand_labels(marker, expand_size)

  coords_t = tuple(data.coordinates.transpose())
  indices = data.indices
  dim = len(coords_t)

  sub_shape = np.broadcast_to(data.crop_size, (dim,))
  view_shape = tuple(marker.shape - sub_shape + 1) + tuple(sub_shape)
  fake_label = as_strided(fake_label, view_shape, fake_label.strides * 2)
  labels = fake_label[coords_t].reshape(len(indices), -1) == indices[:, np.newaxis]
  labels = labels.reshape(len(indices), *sub_shape)

  return labels

def train_with_fake_label(data, model, epochs = 5, batch_size = 256, callback = None):
  try:
    iter(data)
  except TypeError:
    data = (data,)

  model.compile(
      optimizer='Adam',
      loss=tf.losses.BinaryCrossentropy(from_logits=True),
      metrics=['accuracy'],
      )

  patches = [d.patches for d in data]
  patch_labels = [_gen_fake_label(d) for d in data]

  patches_dataset = tf.data.Dataset.from_tensor_slices(np.concatenate(patches))
  patch_labels_dataset = tf.data.Dataset.from_tensor_slices(np.concatenate(patch_labels))
  dataset = tf.data.Dataset.zip((patches_dataset, patch_labels_dataset))

  for epoch in range(epochs):
    print('Epoch : #%i'%epoch)
    model.fit(dataset.batch(batch_size))
    if callback is not None:
      callback(epoch)

  model.summary()

def train_self_supervised(data, model, optimizer = None, n_epochs = 1, area_size = 640, rng = None, steps_per_epoch = 32, callback = None, lam=1.0):
  if rng is None:
    rng = default_rng()

  try:
    iter(data)
  except TypeError:
    data = (data,)

  g = [dd.generator_within_area(rng, area_size=area_size) for dd in data]

  if optimizer is None:
    if not hasattr(model,'optimizer'):
      model.optimizer = tf.keras.optimizers.Adam()
    optimizer = model.optimizer

  for epoch in range(n_epochs):
    loss_t = 0.0
    for _ in range(steps_per_epoch):
      d, c, mask = next(g[rng.integers(len(g))])
      t = int(rng.integers(4))
      d = augment(d,t)
      with tf.GradientTape() as tape:
        y = augment_r(model(d, training = True), t)
        loss = cutter_loss(tf.squeeze(y), c, mask = mask, area_shape=area_size, lam = lam)
      grads = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads,model.trainable_variables))
      loss_t += loss

    loss_t /= steps_per_epoch
    print('Epoch: %i -- loss: %f'%(epoch+1,loss_t))

    if callback is not None:
      callback(epoch, {'loss':loss_t})
