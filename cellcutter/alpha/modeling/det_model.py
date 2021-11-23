''' UNet based cell-detection model '''

import tensorflow as tf
from .layers import *
from .unet import *

class DetModel(tf.keras.Model):
    def __init__(self, n_cls = 3):
        super(DetModel,self).__init__()

        self._n_classes = n_cls
        self._metrics = self._build_metrics()

        self._encoder = UNetEncoder()
        self._decoder = UNetDecoder()
        offset_block = []
        offset_block.append(BatchConv2D(64))
        offset_block.append(BatchConv2D(64))
        offset_block.append(tf.keras.layers.Conv2D(2 * self._n_classes, 3, padding='same'))
        self._offset_block = offset_block

        weight_block = []
        weight_block.append(BatchConv2D(64))
        weight_block.append(BatchConv2D(64))
        weight_block.append(tf.keras.layers.Conv2D(self._n_classes, 3, padding='same'))
        self._weight_block = weight_block

        classification_block = []
        classification_block.append(tf.keras.layers.GlobalAveragePooling2D())
        classification_block.append(tf.keras.layers.Dense(128, activation='relu'))
        classification_block.append(tf.keras.layers.Dense(128, activation='relu'))
        classification_block.append(tf.keras.layers.Dense(self._n_classes, activation='softmax'))
        self._classification_block = classification_block

    def get_config(self):
        return {
            'n_cls': self._n_classes,
        }

    def call(self, inputs, training = None, **kwargs):
        image, labels = inputs
        features = self._encoder(image, training=training)
        outputs = self._decoder(features, training=training)
        x = outputs['0']
        for layer in self._offset_block:
            x = layer(x, training=training)
        offsets = x
        x = outputs['0']
        for layer in self._weight_block:
            x = layer(x, training=training)
        weights = x

        x=features['4']
        for layer in self._classification_block:
            x = layer(x, training=training)
        pred_cls = x

        if training:
            cls = labels['class']
        else:
            cls = tf.argmax(pred_cls, axis=-1, output_type=tf.int32)
        cls = tf.expand_dims(cls, -1)

        offsets = tf.gather(offsets, tf.concat([cls*2, cls*2+1], axis=-1), axis=-1, batch_dims=1)
        weights = tf.gather(weights, cls, axis=-1, batch_dims=1)

        return {
            'offsets': offsets,
            'weights': weights,
            'cls': pred_cls,
        }

    def _build_metrics(self):
        metrics = []
        metric_names = ['model_loss', 'ofs_loss', 'weight_loss', 'classification_loss']
        for name in metric_names:
            metrics.append(tf.keras.metrics.Mean(name, dtype=tf.float32))
        return metrics

    def update_metrics(self, new_metrics):
        logs={}
        for m in self._metrics:
            if m.name in new_metrics:
                m.update_state(new_metrics[m.name])
                logs.update({m.name: m.result()})
        return logs

    @property
    def metrics(self):
        return self._metrics

    def _build_losses(self, labels, model_out):
        gt_offsets = labels['dist_map']
        gt_weights = labels['weights']
        w = gt_weights[...,0]

        gt_cls = tf.expand_dims(labels['class'], -1)
        pred_cls = model_out['cls']
        offsets = model_out['offsets']
        weights = model_out['weights']

        ofs_loss = tf.keras.losses.huber(gt_offsets, offsets, 100.0)
        ofs_loss = tf.reduce_mean(tf.reduce_sum(ofs_loss * w, axis=(1,2))/tf.reduce_sum(w, axis=(1,2)))
        weight_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(gt_weights, weights, from_logits=True))

        classification_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(tf.one_hot(gt_cls[:,0], 3), pred_cls))

        model_loss = 0.02*ofs_loss + weight_loss + classification_loss
        return {
            'model_loss': model_loss,
            'ofs_loss': ofs_loss,
            'weight_loss': weight_loss,
            'classification_loss': classification_loss,
        }

    def train_step(self, inputs):
        _, labels = inputs
        with tf.GradientTape() as tape:
            model_out = self(inputs, training=True)
            losses = self._build_losses(labels, model_out)
        grads = tape.gradient(losses['model_loss'], self.trainable_variables)
        self.optimizer.apply_gradients((g, v) for (g, v) in zip(grads, self.trainable_variables) if g is not None)

        logs = self.update_metrics(losses)
        return logs

    def test_step(self, inputs):
        _, labels = inputs
        model_out = self(inputs, training=True)
        metrics = self._build_losses(labels, model_out)
        return self.update_metrics(metrics)
