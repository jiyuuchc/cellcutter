import sys
from os.path import join
import time
import tifffile
import tensorflow as tf
import numpy as np
import cellcutter
import cellcutter.utils

DATADIR = join('..', 'data')
np.set_printoptions(precision=4)
datafile = 'a1data.npz'
markerfile = 'syn_marker.npz'

print('Loading data...')
data = np.load(join(DATADIR, datafile))
train_data = data['data']
test_id = data['test_id']
test_label = data['test_label']
test_data = (test_id, test_label)

marker_data = np.load(join(DATADIR, markerfile))
print('Loading data... Done')

print('Segmentation - FL image')
# generate a fake datase -- becasue our validation data are based on old markers
train_img = np.stack((train_data[..., 0], marker_data['fl_img']), axis = -1)
old_marker = train_data[...,3]
fake_dataset = cellcutter.Dataset(train_img, old_marker)
test_img = fake_dataset.patches[test_id, ...]

real_marker = marker_data['fl_marker']
dataset = cellcutter.Dataset(train_img, real_marker, mask_img = train_data[...,4])

model = cellcutter.UNet4(bn=True)
start = time.time()
cellcutter.train_self_supervised(dataset, model, n_epochs = 50, val_data=(test_img, test_label))
print('Elapsed time: %f'%(time.time() - start))
#pred = tf.sigmoid(model(dataset.patches)).numpy().squeeze() > 0.5

tifffile.imwrite('label-FL-syn.tif', cellcutter.utils.draw_label(dataset, model, np.zeros_like(train_data[...,0])))
tifffile.imwrite('border-FL-syn.tif', cellcutter.utils.draw_border(dataset, model, np.zeros_like(train_data[...,0])))

print('Segmentation - BF image')
# generate a fake datase -- becasue our validation data are based on old markers
train_img = np.stack((train_data[..., 1], marker_data['bf_img']), axis = -1)
old_marker = train_data[...,3]
fake_dataset = cellcutter.Dataset(train_img, old_marker)
test_img = fake_dataset.patches[test_id, ...]

real_marker = marker_data['bf_marker']
dataset = cellcutter.Dataset(train_img, real_marker, mask_img = train_data[...,5])

model = cellcutter.UNet4(bn=True)
start = time.time()
cellcutter.train_self_supervised(dataset, model, n_epochs = 50, val_data=(test_img, test_label))
print('Elapsed time: %f'%(time.time() - start))
#pred = tf.sigmoid(model(dataset.patches)).numpy().squeeze() > 0.5

tifffile.imwrite('label-BF-syn.tif', cellcutter.utils.draw_label(dataset, model, np.zeros_like(train_data[...,0])))
tifffile.imwrite('border-BF-syn.tif', cellcutter.utils.draw_border(dataset, model, np.zeros_like(train_data[...,0])))
