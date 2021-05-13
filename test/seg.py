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

def train(train_dataset, test_data, epochs=50):
  start = time.time()
  model = cellcutter.UNet4(bn=True)
  test_id, test_label = test_data
  test_img = dataset.patches[test_id,...]
  cellcutter.train_self_supervised(dataset, model, n_epochs = epochs, val_data=(test_img, test_label))
  print('Elapsed time: %f'%(time.time() - start))
  pred = tf.sigmoid(model(dataset.patches)).numpy().squeeze() > 0.5
  return model, pred

print('Loading data...')
data = np.load(join(DATADIR, datafile))
train_data = data['data']
test_id = data['test_id']
test_label = data['test_label']
test_data = (test_id, test_label)
print('Loading data... Done')

print('Segmentation - FL image')
dataset = cellcutter.Dataset(train_data[...,0], train_data[...,3], mask_img = train_data[...,4])
model, pred_fl = train(dataset, test_data)
tifffile.imwrite('label-FL.tif', cellcutter.utils.draw_label(dataset, model, np.zeros_like(train_data[...,0])))
tifffile.imwrite('border-FL.tif', cellcutter.utils.draw_border(dataset, model, np.zeros_like(train_data[...,0])))

print('Segmentation - FL + Nuc images')
dataset = cellcutter.Dataset(train_data[...,(0,2)], train_data[...,3], mask_img = train_data[...,4])
model, pred_flnuc = train(dataset, test_data)
tifffile.imwrite('label-FLNuc.tif', cellcutter.utils.draw_label(dataset, model, np.zeros_like(train_data[...,0])))
tifffile.imwrite('border-FLNuc.tif', cellcutter.utils.draw_border(dataset, model, np.zeros_like(train_data[...,0])))

print('Segmentation - BF images')
dataset = cellcutter.Dataset(train_data[...,1], train_data[...,3], mask_img = train_data[...,5])
model, pred_bf = train(dataset, test_data)
tifffile.imwrite('label-BF.tif', cellcutter.utils.draw_label(dataset, model, np.zeros_like(train_data[...,0])))
tifffile.imwrite('border-BF.tif', cellcutter.utils.draw_border(dataset, model, np.zeros_like(train_data[...,0])))

print('Segmentation - BF + Nuc images')
dataset = cellcutter.Dataset(train_data[..., (1,2)], train_data[...,3], mask_img = train_data[...,5])
model, pred_bfnuc = train(dataset, test_data)
tifffile.imwrite('label-BFNuc.tif', cellcutter.utils.draw_label(dataset, model, np.zeros_like(train_data[...,0])))
tifffile.imwrite('border-BFNuc.tif', cellcutter.utils.draw_border(dataset, model, np.zeros_like(train_data[...,0])))

print('Segmentation - FL + BF images')
dataset = cellcutter.Dataset(train_data[..., (0,1)], train_data[...,3], mask_img = train_data[...,4])
model, pred_flbf = train(dataset, test_data)
tifffile.imwrite('label-FLBF.tif', cellcutter.utils.draw_label(dataset, model, np.zeros_like(train_data[...,0])))
tifffile.imwrite('border-FLBF.tif', cellcutter.utils.draw_border(dataset, model, np.zeros_like(train_data[...,0])))

print('Segmentation - FL + BF + Nuc images')
dataset = cellcutter.Dataset(train_data[..., (0,1,2)], train_data[...,3], mask_img = train_data[...,4])
model, pred_flbfnuc = train(dataset, test_data)
tifffile.imwrite('label-FLBFNuc.tif', cellcutter.utils.draw_label(dataset, model, np.zeros_like(train_data[...,0])))
tifffile.imwrite('border-FLBFNuc.tif', cellcutter.utils.draw_border(dataset, model, np.zeros_like(train_data[...,0])))

def miou(label, pred, n = 2):
  v = 0
  for l in range(n):
    intersect = np.sum((pred == l) * (label == l), axis=(1,2))
    union = np.sum((pred==l), axis=(1,2)) + np.sum((label==l), axis=(1,2)) - intersect
    v += np.mean(intersect / (union + 0.01))
  return v / n

preds = np.stack((pred_fl, pred_bf, pred_flbf, pred_flnuc, pred_bfnuc, pred_flbfnuc))
mious_mat = np.ones((6,6))
for i in range(5):
  for j in range(i+1, 6):
    mious_mat[i,j] = mious_mat[j,i] = miou(preds[i,...], preds[j,...])

print(mious_mat)
