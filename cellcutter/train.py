import tensorflow as tf
import numpy as np

def augment(img, t):
  if t == 1 or t == 3:
    img = tf.image.flip_left_right(img)
  if t == 2 or t == 3:
    img = tf.image.flip_up_down(img)
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
