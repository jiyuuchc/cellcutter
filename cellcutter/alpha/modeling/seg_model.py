''' implementaion of the segmentation model '''
import tensorflow as tf
import tensorflow.keras.layers as layers
from .common import *
from .unet import *

class SegModel(tf.keras.Model):
    def __init__(self, encoder, decoder, mask_layer):
        super(SegModel,self).__init__()

        self._config_dict = {
            'encoder': encoder,
            'decoder': decoder,
            'mask_layer': mask_layer,
        }
        self._encoder = encoder
        self._decoder = decoder
        self._mask_layer = mask_layer
        self._build_metrics()

    def get_config(self):
        return self._config_dict

    def _build_metrics(self):
        metrics = []
        metric_names = ['model_loss', 'mask_loss',]
        for name in metric_names:
            metrics.append(tf.keras.metrics.Mean(name, dtype=tf.float32))
        self._metrics = metrics

    def _update_metrics(self, new_metrics):
        logs={}
        for m in self._metrics:
            if m.name in new_metrics:
                m.update_state(new_metrics[m.name])
                logs.update({m.name: m.result()})
        return logs

    @property
    def metrics(self):
        return self._metrics

    def _adjust_bbox(self, bbox):
        '''
        adjust bbox size to be 2x larger
        '''
        row_starts = bbox.row_starts()
        bbox = bbox.merge_dims(0,1)
        cy = (bbox[...,0] + bbox[...,2])/2
        cx = (bbox[...,1] + bbox[...,3])/2
        hh = (bbox[...,2] - bbox[...,0])*1.5
        ww = (bbox[...,3] - bbox[...,1])*1.5
        bbox_out = tf.stack([
              cy - hh / 2,
              cx - ww / 2,
              cy + hh / 2,
              cx + ww / 2,
          ], axis = -1)
        bbox_out = tf.RaggedTensor.from_row_starts(bbox_out, row_starts)
        return tf.cast(bbox_out, tf.float32)

    def call(self, inputs, training = None):
        (imgs, bboxes, scores, cls), labels = inputs
        features = self._encoder(imgs, training=training)
        outputs = self._decoder(features, training=training)
        data = outputs, bboxes, scores, cls
        return self._mask_layer((data, labels), training=training)

    def train_step(self, inputs):
        data, labels = inputs
        with tf.GradientTape() as tape:
            model_out = self(inputs, training=True)
            mask_loss = self._mask_layer._compute_losses(labels, model_out)
        grads = tape.gradient(mask_loss, self.trainable_variables)
        self.optimizer.apply_gradients((g, v) for (g, v) in zip(grads, self.trainable_variables) if g is not None)

        logs = self._update_metrics({'mask_loss': mask_loss})
        return logs

    def test_step(self, inputs):
        data, labels = inputs
        model_out = self(inputs, training=False)
        mask_loss = self._mask_layer._compute_losses(labels, model_out)
        logs = self._update_metrics({'mask_loss': mask_loss})
        return logs

    def predict(self, inputs):
        imgs, bboxes, scores, cls = inputs
        features = self._encoder(imgs, training=False)
        outputs = self._decoder(features, training=False)
        data = outputs, bboxes, scores, cls
        return self._mask_layer.predict(data)
