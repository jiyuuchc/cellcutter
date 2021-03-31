import tensorflow as tf
import numpy as np
from numpy.random import default_rng
from .loss import cutter_loss

def augment(img, t):
  if t & 1:
    img = tf.image.flip_left_right(img)
  if t & 2:
    img = tf.image.flip_up_down(img)
  if t & 4:
    img = tf.image.transpose(img)
  return img

def train_with_label(data, model, epochs = 1, batch_size = 256):
  model.compile(
      optimizer='Adam',
      loss=tf.losses.BinaryCrossentropy(from_logits=True),
      metrics=['accuracy'],
      )
  dataset = data.tf_dataset_with_label()

  for epoch in range(epochs):
    print('Epoch : #%i'%epoch)
    model.fit(dataset.batch(batch_size))

  model.summary()

def train_self_supervised(data, model, optimizer = None, n_epochs = 1, area_size = 640, rng = None, batch_size = 32, callback = None):
  if rng is None:
    rng = default_rng()
  g = data.generator_within_area(rng, area_size=area_size)
  if optimizer is None:
    if not hasattr(model,'optimizer'):
      model.optimizer = tf.keras.optimizers.Adam()
    optimizer = model.optimizer

  for epoch in range(n_epochs):
    loss_t = 0.0
    for _ in range(batch_size):
      d, c, mask = next(g)
      t = int(rng.integers(4))
      d = augment(d,t)
      with tf.GradientTape() as tape:
        y = augment(model(d, training = True), t)
        loss = cutter_loss(tf.squeeze(y), c, mask = mask, area_size=area_size)
      grads = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads,model.trainable_variables))
      loss_t += loss

    loss_t /= batch_size
    print('Epoch: %i -- loss: %f'%(epoch+1,loss_t))

    if callback is not None:
      callback(loss_t)
