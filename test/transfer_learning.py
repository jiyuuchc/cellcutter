from os.path import join
import time
import tifffile
import tensorflow as tf
import numpy as np
import cellcutter
import cellcutter.utils

def miou(label, pred, n = 2):
  v = 0
  for l in range(n):
    intersect = np.sum((pred == l) * (label == l), axis=(1,2))
    union = np.sum((pred==l), axis=(1,2)) + np.sum((label==l), axis=(1,2)) - intersect
    v += (intersect / (union + 0.01)).mean()
  return v / n

def train(train_dataset, epochs=50):
  start = time.time()
  model = cellcutter.UNet4(bn=True)
  cellcutter.train_self_supervised(dataset, model, n_epochs = epochs)
  print('Elapsed time: %f'%(time.time() - start))
  return model

DATADIR = join(PROJ, 'data')
np.set_printoptions(precision=4)

files = ['a1data.npz', 'a2data.npz', 'a3data.npz']
data = [np.load(join(DATADIR, f))['data'] for f in files]

print('Training three models with FL data')
dataset = [cellcutter.Dataset(d[...,(0,2)], d[...,3], mask_img = d[...,4]) for d in data]
models = [train(ds) for ds in dataset]
preds = [m(ds.patches).numpy().squeeze() > 0.5 for m,ds in zip(models, dataset)]

for i, m in enumerate(models):
  print('Model #%i :'%(i))
  for j,ds in enumerate(dataset):
     new_pred = m(ds.patches).numpy().squeeze() > 0.5
     v = miou(preds[j], new_pred)
     print('\tmIOU against dataset %i: %f'%(j, v))

print('Training three models with BF data')
dataset = [cellcutter.Dataset(d[...,(1,2)], d[...,3], mask_img = d[...,4]) for d in data]
models = [train(ds) for ds in dataset]
preds = [ m(ds.patches).numpy().squeeze() > 0.5 for m,ds in zip(models, dataset)]

for i, m in enumerate(models):
  print('Model #%i :'%(i))
  for j,ds in enumerate(dataset):
     new_pred = m(ds.patches).numpy().squeeze() > 0.5
     v = miou(preds[j], new_pred)
     print('\tmIOU against dataset %i: %f'%(j, v))
