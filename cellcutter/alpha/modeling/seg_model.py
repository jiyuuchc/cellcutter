''' implementaion of the segmentation model '''
import tensorflow as tf
import tensorflow.keras.layers as layers
from .common import *
from .unet import *

class SegModel(tf.keras.Model):
    def __init__(self, encoder, decoder, n_cls = 3, crop_size=64, iou_low_threshold=.3):
        super(SegModel,self).__init__()

        self._config_dict = {
            'encoder': encoder,
            'decoder': decoder,
            'n_cls': n_cls,
            'crop_size': crop_size,
            'iou_low_threshold': iou_low_threshold,
        }
        self._encoder = encoder
        self._decoder = decoder

        self._build_metrics()

        conv_block = [
            BatchConv2D(64, name='out_conv1'),
            BatchConv2D(64, name='out_conv2'),
        ]
        self._conv_block = conv_block
        self._segment = layers.Conv2D(self._config_dict['n_cls'], 3, padding='same', name='segment')
        self._mixing = layers.Conv2D(1, 1, name='mixing')

        regression_block = [
            layers.MaxPool2D((4,4), name='regression_pooling'),
            layers.Flatten(name='regression_flatten'),
            layers.Dense(512, activation='relu', name='regression_dense1'),
            layers.Dense(512, activation='relu', name='regression_dense2'),
            layers.Dense(self._config_dict['n_cls'], name='regression_out'),
        ]
        self._regression_block = regression_block

    def get_config(self):
        return self._config_dict

    def _build_metrics(self):
        metrics = []
        metric_names = ['model_loss', 'mask_loss', 'regression_loss',]
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

    # def _binary_iou(self, gt_masks, masks):
    #     gt_mask_b = gt_masks > 0.5
    #     mask_b = masks > 0.0
    #     intersec = tf.math.count_nonzero(tf.logical_and(gt_mask_b, mask_b), axis=(1,2,3))
    #     union = tf.math.count_nonzero(tf.logical_or(gt_mask_b, mask_b), axis=(1,2,3))
    #     return intersec/union

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

    def do_call(self, inputs, training = None):
        data, labels = inputs
        imgs = data['image']
        features = self._encoder(imgs, training=training)
        outputs = self._decoder(features, training=training)

        bboxes = data['proposal_bboxes']
        crop_size = self._config_dict['crop_size']
        cls = data['class']
        cls_repeat = tf.repeat(cls, bboxes.row_lengths())
        actual_bboxes = self._adjust_bbox(bboxes)
        feature_crops = crop_features(outputs['0'], actual_bboxes, crop_size)
        proposals = tf.ensure_shape(data['proposal'], (None, 544,704))
        proposal_crops = crop_proposals(proposals, actual_bboxes, crop_size)

        x = feature_crops
        for layer in self._conv_block:
            x = layer(x, training=training)
        pre_masks = self._segment(x)
        pre_masks = tf.gather(pre_masks, cls_repeat[:,None], axis=-1, batch_dims=1)
        masks = self._mixing(tf.concat([pre_masks, proposal_crops], axis=-1), training=training)

        for layer in self._regression_block:
            x = layer(x, training=training)
        regression_out = tf.gather(x, cls_repeat[:,None], axis=-1, batch_dims=1)

        return {
            'masks': masks,
            'regressions': regression_out,
            'actual_bboxes': actual_bboxes,
        }

    def _compute_losses(self, labels, model_out):
        masks = model_out['masks']
        _,height,width,_ = labels['weights'].get_shape()
        crop_size = self._config_dict['crop_size']
        gt_masks = crop_masks(
            labels['mask_indices'],
            model_out['actual_bboxes'],
            labels['matched_indices'],
            crop_size, height, width,
        )

        regression_out = model_out['regressions'],
        proposal_ious = labels['matched_ious'].merge_dims(0,1)
        regression_loss = tf.reduce_mean(
            tf.keras.losses.huber(proposal_ious, regression_out, 0.5)
        )

        gt_masks = tf.cast(gt_masks>.5, tf.float32)
        #mask_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(gt_masks, masks, from_logits=True), axis=(1,2))
        mask_loss = tf.reduce_mean(tfa.losses.sigmoid_focal_crossentropy(gt_masks, masks, from_logits=True), axis=(1,2))
        pos_indices = tf.where(proposal_ious >= self._config_dict['iou_low_threshold'])
        mask_loss = tf.reduce_mean(tf.gather(mask_loss, pos_indices))

        model_loss = mask_loss + regression_loss
        return {
            'model_loss': model_loss,
            'mask_loss': mask_loss,
            'regression_loss': regression_loss,
        }

    def train_step(self, inputs):
        data, labels = inputs
        with tf.GradientTape() as tape:
            model_out = self.do_call(inputs, training=True)
            losses = self._compute_losses(labels, model_out)
        grads = tape.gradient(losses['model_loss'], self.trainable_variables)
        self.optimizer.apply_gradients((g, v) for (g, v) in zip(grads, self.trainable_variables) if g is not None)

        logs = self._update_metrics(losses)
        return logs

    def test_step(self, inputs):
        data, labels = inputs
        model_out = self.do_call(inputs, training=False)
        losses = self._compute_losses(labels, model_out)
        logs = self._update_metrics(losses)

        return logs
