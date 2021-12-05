import tensorflow as tf
from .layers import *
from .unet import *
from ..ops import *

class MaskModel(tf.keras.Model):
    def __init__(self, encoder, decoder, n_cls = 3, crop_size=16, min_iou = 0.3, with_masks=False):
        super(MaskModel, self).__init__()
        self._metrics = self._build_metrics()

        self._encoder = encoder
        self._decoder = decoder
        self._n_classes = n_cls
        self._crop_size = crop_size
        self._min_iou = min_iou
        self._with_masks = with_masks

        proposal_block = []
        proposal_block.append(BatchConv2D(128, strides=2))
        proposal_block.append(BatchConv2D(128, strides=2))
        proposal_block.append(tf.keras.layers.Flatten())
        proposal_block.append(tf.keras.layers.Dense(512, activation='relu'))
        proposal_block.append(tf.keras.layers.Dense(512, activation='relu'))
        self._proposal_block = proposal_block
        self._proposal_regression = tf.keras.layers.Dense(4 * n_cls)
        self._proposal_score = tf.keras.layers.Dense(n_cls)

        mask_block = []
        for k in range(n_cls):
            block = []
            block.append(BatchConv2D(128))
            block.append(BatchConv2D(128))
            block.append(tf.keras.layers.Conv2D(n_cls, 1))
            mask_block.append(block)
        self._mask_block = mask_block

    def get_config(self):
        return {
            'encoder': self._encoder,
            'decoder': self._decoder,
            'n_cls': self._n_classes,
            'crop_size': self._crop_size,
            'min_iou': self._min_iou,
            'with_masks': self._with_masks,
        }

    def _build_metrics(self):
        metrics = []
        metric_names = [
            'model_loss', 'regression_score_loss', 'regression_loss', 'mask_loss', 'a50', 'a75', 'a90',
        ]
        for name in metric_names:
            metrics.append(tf.keras.metrics.Mean(name, dtype=tf.float32))
        return metrics

    @property
    def metrics(self):
        return self._metrics

    def update_metrics(self, new_metrics):
        logs={}
        for m in self._metrics:
            if m.name in new_metrics:
                m.update_state(new_metrics[m.name])
                logs.update({m.name: m.result()})
        return logs

    def call(self, inputs, training = None, **kwargs):
        data, labels = inputs
        images = data['image']
        _,h,w,_ = images.get_shape()
        features = self._encoder(images, training=training)
        outputs = self._decoder(features, training=training)

        cls = data['class']
        bboxes = data['proposal_bboxes']
        feature_crops = crop_features(outputs['0'], bboxes, self._crop_size)

        proposal_crops = crop_proposals(data['proposal'], bboxes, self._crop_size)
        # proposal_crops = tf.ensure_shape(proposal_crops, [None, self._crop_size, self._crop_size, 1])
        x = feature_crops * proposal_crops

        for layer in self._proposal_block:
            x = layer(x)
        cls_repeat = tf.repeat(cls, bboxes.row_lengths())
        regression_out = self._proposal_regression(x)
        regression_out = tf.gather(regression_out, tf.stack([cls_repeat*4, cls_repeat*4+1, cls_repeat*4+2, cls_repeat*4+3], axis=-1), axis=-1, batch_dims=1)
        regression_score = self._proposal_score(x)
        regression_score = tf.gather(regression_score, cls_repeat[:, None], axis=-1, batch_dims=1)
        regression_bboxes = recover_boxes(bboxes.merge_dims(0,1), regression_out)

        model_outputs = {
            'regression_scores': regression_score,
            'regressions': regression_out,
            'regression_bboxes': regression_bboxes,
            }

        if not self._with_masks:
            return model_outputs

        feature_crops = crop_features(outputs['0'], regression_bboxes, 64)
        x = feature_crops
        for layer in self._mask_block[proposals['class']]:
            x=layer(x)
        model_outputs.update({'mask_out': x,})

        return model_outputs

    def _build_losses(self, inputs, model_out):
        data, labels = inputs
        gt_scores = labels['matched_ious'].merge_dims(0,1)
        scores = model_out['regression_scores'][:,0]
        regression_score_loss = tf.reduce_mean(tf.keras.losses.huber(gt_scores, scores, 0.5))

        pos_indices = tf.where(gt_scores > self._min_iou)
        proposal_bboxes = tf.gather(data['proposal_bboxes'].merge_dims(0,1), pos_indices)
        matched_bboxes = tf.gather(labels['matched_bboxes'].merge_dims(0,1), pos_indices)
        regression_targets = compare_boxes(proposal_bboxes, matched_bboxes)
        regressions = tf.gather(model_out['regressions'], pos_indices)
        regression_loss = tf.reduce_mean(tf.keras.losses.huber(regression_targets, regressions, 1))

        model_loss = regression_score_loss + regression_loss
        losses = {
            'model_loss': model_loss,
            'regression_score_loss': regression_score_loss,
            'regression_loss': regression_loss,
        }
        if not self._with_masks:
            return losses

        _, height, width, _ = data['image'].get_shape()
        matched_indices = labels['matched_indices']
        regression_bboxes = tf.RaggedTensor.from_row_starts(model_out['regression_bboxes'], matched_indices.row_starts())
        gt_mask_crops = crop_masks(lables['mask_indices'], regression_bboxes, matched_indices, 64, height, width)
        gt_mask_crops = tf.gather(gt_mask_crops, pos_indices)
        mask_crops = model_out('mask_out')
        mask_crops = tf.gather(mask_crops, pos_indices)
        mask_loss = tf.recude_mean(tf.keras.losses.binary_crossentropy(gt_mask_crops, mask_crops, from_logits=True))
        model_loss += mask_loss

        losses.update({
            'model_loss': model_loss,
            'mask_loss': mask_loss,
        })
        return losses

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            model_out = self(inputs, training=True)
            losses = self._build_losses(inputs, model_out)
        grads = tape.gradient(losses['model_loss'], self.trainable_variables)
        self.optimizer.apply_gradients((g, v) for (g, v) in zip(grads, self.trainable_variables) if g is not None)
        logs = self.update_metrics(losses)
        return logs

    def test_step(self, inputs):
        data, labels = inputs
        model_out = self(inputs, training=False)
        metrics = self._build_losses(inputs, model_out)

        n_gt_masks = tf.reduce_sum(labels['bboxes'].row_lengths())
        gt_regression_bboxes = labels['matched_bboxes'].merge_dims(0,1)
        regression_bboxes = model_out['regression_bboxes']
        new_ious= box_ious(regression_bboxes, gt_regression_bboxes)
        metrics.update({
            'a50': tf.math.count_nonzero(new_ious>0.5) / n_gt_masks,
            'a75': tf.math.count_nonzero(new_ious>0.75) / n_gt_masks,
            'a90': tf.math.count_nonzero(new_ious>0.90) / n_gt_masks,
            # 'r50': tf.math.count_nonzero(m_ious_out>0.5) / n_gt_masks,
            # 'r75': tf.math.count_nonzero(m_ious_out>0.75) / n_gt_masks,
            # 'r95': tf.math.count_nonzero(m_ious_out>0.95) / n_gt_masks,
        })
        logs =  self.update_metrics(metrics)
        return logs
