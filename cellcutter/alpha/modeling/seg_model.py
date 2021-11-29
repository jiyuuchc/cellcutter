''' implementaion of the segmentation model '''
import tensorflow as tf
from .layers import *
from .unet import *

class SegModel(tf.keras.Model):
    def __init__(self, encoder, decoder, n_cls = 3):
        super(SegModel,self).__init__()
        self._metrics = self._build_metrics()
        self._n_classes = n_cls + 1

        self._encoder = encoder
        self._decoder = decoder

        seg_block = []
        seg_block.append(BatchConv2D(64))
        seg_block.append(tf.keras.layers.Conv2D(self._n_classes, 3, padding='same'))
        self._seg_block = seg_block

        classification_block = []
        classification_block.append(tf.keras.layers.GlobalAveragePooling2D())
        classification_block.append(tf.keras.layers.Dense(128, activation='relu'))
        classification_block.append(tf.keras.layers.Dense(128, activation='relu'))
        classification_block.append(tf.keras.layers.Dense(self._n_classes, activation='softmax'))
        self._classification_block = classification_block

    def get_config(self):
        return {
            'encoder': self._encoder,
            'decoder': self._decoder,
            'n_cls': self._n_classes,
        }

    def call(self, inputs, training = None, **kwargs):
        src_imgs = inputs['source_image']
        preds = inputs['proposal']
        gt_cls = inputs['gt_cell_type']

        #features = self._encoder(tf.concat([src_imgs, preds], axis=-1), training=training)
        features = self._encoder(src_imgs, training=training)
        outputs = self._decoder(features, training=training)

        x = outputs['0']
        for layer in self._seg_block:
            x = layer(x, training=training)
            x = x + preds
        mask = tf.gather(x, gt_cls[:,None], axis=-1, batch_dims=1)

        x=features['4']
        for layer in self._classification_block:
            x = layer(x, training=training)
        pred_cls = x

        return {
            'mask': mask,
            'cls': pred_cls,
        }

    def _build_metrics(self):
        metrics = []
        metric_names = ['model_loss', 'mask_loss', 'classification_loss', 'iou', 'cls_acc', 'cls_recall', 'iou50', 'iou75', 'iou95']
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

    def _build_losses(self, inputs, model_out):
        gt_cls = inputs['gt_cell_type']
        pred_cls = model_out['cls']
        gt_mask = inputs['gt_mask']
        pred_mask = model_out['mask']

        cell_ind = tf.where(gt_cls)
        gt_mask = tf.gather(gt_mask, cell_ind)
        pred_mask = tf.gather(pred_mask, cell_ind)

        mask_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(gt_mask, pred_mask, from_logits=True))
        classification_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(tf.one_hot(gt_cls, 4), pred_cls))

        model_loss = mask_loss + classification_loss
        return {
            'model_loss': model_loss,
            'mask_loss': mask_loss,
            'classification_loss': classification_loss,
        }

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            model_out = self(inputs, training=True)
            losses = self._build_losses(inputs, model_out)
        grads = tape.gradient(losses['model_loss'], self.trainable_variables)
        self.optimizer.apply_gradients((g, v) for (g, v) in zip(grads, self.trainable_variables) if g is not None)

        logs = self.update_metrics(losses)
        return logs

    def _binary_iou(self, gt_masks, masks):
        gt_mask_b = gt_masks > 0.5
        mask_b = masks > 0.0
        intersec = tf.math.count_nonzero(tf.logical_and(gt_mask_b, mask_b), axis=(1,2,3))
        union = tf.math.count_nonzero(tf.logical_or(gt_mask_b, mask_b), axis=(1,2,3))
        return intersec/union

    def test_step(self, inputs):
        model_out = self(inputs, training=False)
        metrics = self._build_losses(inputs, model_out)

        iou = self._binary_iou(inputs['gt_mask'], model_out['mask'])
        iou = tf.gather(iou, tf.where(inputs['gt_cell_type']))

        cls = tf.argmax(model_out['cls'],axis=-1)
        correct_cls = tf.math.count_nonzero(tf.logical_and(inputs['gt_cell_type']>0, cls > 0))
        metrics.update({
            'iou': iou,
            'iou50': tf.cast(iou>0.5, dtype=tf.float32),
            'iou75': tf.cast(iou>0.75, dtype=tf.float32),
            'iou95': tf.cast(iou>0.95, dtype=tf.float32),
            'cls_acc': correct_cls / tf.math.count_nonzero(cls>0),
            'cls_recall': correct_cls / tf.math.count_nonzero(inputs['gt_cell_type']>0),
        })
        return self.update_metrics(metrics)
