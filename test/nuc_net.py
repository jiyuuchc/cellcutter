import sys
from os.path import join
import time
import tifffile
import tensorflow as tf
import numpy as np
import cellcutter
import cellcutter.utils

# train a unet to map fluorescence image to nucleus Image

DATADIR=join('..', 'data')
np.set_printoptions(precision=4)
datafiles = ['a3data.npz', 'a2data.npz', 'a1data.npz'] # use the first two for training

imgsize = 1760

def label2binary(label):
  from skimage.segmentation import find_boundaries
  boundry = find_boundaries(label)
  mask = label.copy()
  mask[label > 0] = 1
  mask[boundry] = 0
  return mask.astype(np.int)

def normalize_img(img):
  img = img.astype(np.float32)
  img -= img.mean(axis = (-1,-2), keepdims = True)
  img /= img.std(axis = (-1,-2), keepdims = True)
  return img

def get_img_label_pair(data, use_fl = True):
  imgs = data['data']
  if use_fl:
    img = imgs[..., 0] # FL imag
  else:
    img = imgs[..., 1] # BF img

  img = normalize_img(img)
  label = label2binary(imgs[..., 3]) # revert the segmented nucleus images back to simple binary
  return (img, label)

print('Loading data...')

alldata = [get_img_label_pair(np.load(join(DATADIR, datafile)), True) for datafile in datafiles]
data, label = zip(*alldata)

data = tf.stack(data)
label = tf.stack(label)
data = data[..., None] #tf.imate requires last dimension to be channel
label = label[..., None]
data = tf.image.resize_with_pad(data, imgsize, imgsize)  # the unet requires image size to be multiple of 16
label = tf.image.resize_with_pad(label, imgsize, imgsize)

train_data = data[:-1, ...]
train_label = label[:-1, ...]

test_data = data[-1, ...]
test_label = label[-1, ...]
test_data = test_data[None, ...]
test_label = test_label[None, ...]

print('Loading data... Done')

train_data = tf.concat((train_data,
                        tf.image.flip_left_right(train_data),
                        tf.image.flip_up_down(train_data),
                        tf.image.flip_up_down(tf.image.flip_left_right(train_data))), axis = 0)
train_label = tf.concat((train_label,
                        tf.image.flip_left_right(train_label),
                        tf.image.flip_up_down(train_label),
                        tf.image.flip_up_down(tf.image.flip_left_right(train_label))), axis = 0)

#split into 4 to avoid memory issues
train_data = tf.reshape(train_data, [-1, imgsize, 2, imgsize // 2, 1])
train_data = tf.transpose(train_data, [0, 2, 1, 3, 4])
train_data = tf.reshape(train_data, [-1, imgsize // 2 , imgsize // 2, 1])
train_label = tf.reshape(train_label, [-1, imgsize, 2, imgsize // 2, 1])
train_label = tf.transpose(train_label, [0, 2, 1, 3, 4])
train_label = tf.reshape(train_label, [-1, imgsize // 2, imgsize // 2, 1])

train_label = train_label[..., 0] #dont' need the channel dimension now
test_label = test_label[..., 0]

model = cellcutter.UNet4()
model.compile(
    optimizer = 'Adam',
    loss = tf.losses.BinaryCrossentropy(from_logits=True),
    metrics = ['accuracy']
)
model.fit(train_data, train_label, batch_size=1, epochs=10, validation_batch_size=1, validation_data=(test_data, test_label))

pred = tf.sigmoid(model(test_data)).numpy().squeeze()
pred = pred[5:-5,5:-5]
tifffile.imwrite('syn_nuc.tif', pred)
