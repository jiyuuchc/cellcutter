''' UNet based cell-detection model '''

import tensorflow as tf
from .layers import *
from .unet import *
from ..ops import *

def _compare_boxes(boxes, gt_boxes):
    rr = (boxes[:,:,0] + boxes[:,:,2])/2
    cc = (boxes[:,:,1] + boxes[:,:,3])/2
    hh = boxes[:,:,2] - boxes[:,:,0]
    ww = boxes[:,:,3] - boxes[:,:,1]

    grr = (gt_boxes[:,:,0] + gt_boxes[:,:,2])/2
    gcc = (gt_boxes[:,:,1] + gt_boxes[:,:,3])/2
    ghh = gt_boxes[:,:,2] - gt_boxes[:,:,0]
    gww = gt_boxes[:,:,3] - gt_boxes[:,:,1]

    drr = (grr - rr) / hh
    dcc = (gcc - cc) / ww
    dhh = (ghh - hh) / hh
    dww = (gww - ww) / ww
    return tf.stack([drr, dcc, dhh, dww], axis=-1)

def _recover_boxes(boxes, regression_out):
    rr = (boxes[...,0] + boxes[...,2])/2
    cc = (boxes[...,1] + boxes[...,3])/2
    hh = boxes[...,2] - boxes[...,0]
    ww = boxes[...,3] - boxes[...,1]
    rr += regression_out[...,0] * hh
    cc += regression_out[...,1] * ww
    hh += regression_out[...,2] * hh
    ww += regression_out[...,3] * ww
    r0 = rr - hh / 2.
    c0 = cc - ww / 2.
    r1 = rr + hh / 2.
    c1 = cc + ww / 2.
    return tf.stack([r0,c0,r1,c1], axis=-1)

def _eval_boxes(box_out, gt_boxes):
    matched_boxes, matched_indices, m_ious, _ = box_matching(box_out, gt_bboxes)

class DetModel(tf.keras.Model):
    def __init__(
        self,
        encoder,
        decoder,
        n_cls = 3,
        eps = 1.0,
        crop_size=16,
        min_samples = 4.0,
        negative_iou=0.1,
        positive_iou=0.5
    ):
        super(DetModel,self).__init__()

        self.burn_in_is_done = False

        self._n_classes = n_cls
        self._eps = eps
        self._min_samples = min_samples
        self._crop_size = crop_size
        self._negative_iou = negative_iou
        self._positive_iou = positive_iou

        self._metrics = self._build_metrics()

        self._encoder = encoder
        self._decoder = decoder
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

        proposal_block = []
        proposal_block.append(BatchConv2D(128, strides=2))
        proposal_block.append(tf.keras.layers.Flatten())
        proposal_block.append(tf.keras.layers.Dense(512, activation='relu'))
        proposal_block.append(tf.keras.layers.Dense(512, activation='relu'))
        self._proposal_block = proposal_block
        self._proposal_regression = tf.keras.layers.Dense(4 * n_cls)
        self._proposal_score = tf.keras.layers.Dense(n_cls)

    def get_config(self):
        return {
            'encoder': self._encoder,
            'decoder': self._decoder,
            'n_cls': self._n_classes,
            'eps': self._eps,
            'min_samples': self._min_samples,
            'crop_size': self._crop_size,
        }

    def call(self, inputs, training = None, **kwargs):
        image, labels = inputs
        h = tf.shape(image)[1]
        w = tf.shape(image)[2]
        hwhw = tf.cast([h,w,h,w], tf.float32)

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

        if training and labels is not None:
            cls = labels['class']
        else:
            cls = tf.argmax(pred_cls, axis=-1, output_type=tf.int32)
        #cls = tf.expand_dims(cls, -1)

        offsets = tf.gather(offsets, tf.stack([cls*2, cls*2+1], axis=-1), axis=-1, batch_dims=1)
        weights = tf.gather(weights, cls[:,None], axis=-1, batch_dims=1)

        model_outputs = {
            'offsets': offsets,
            'weights': weights,
            'cls': pred_cls,
        }

        if self.burn_in_is_done:
            proposals = pred_labels(offsets, weights, eps=self._eps, min_samples=self._min_samples)
            bboxes = tf.cast(bbox_of_proposals(proposals), tf.float32) / hwhw
            x = tf.image.crop_and_resize(
                outputs['1'],
                bboxes.merge_dims(0,1),
                tf.repeat(tf.range(bboxes.nrows(), dtype=tf.int32), bboxes.row_lengths()),
                [self._crop_size, self._crop_size]
            )
            for layer in self._proposal_block:
                x = layer(x)

            cls_repeat = tf.repeat(cls, bboxes.row_lengths())
            regression_out = self._proposal_regression(x)
            regression_out = tf.gather(regression_out, tf.stack([cls_repeat*4, cls_repeat*4+1, cls_repeat*4+2, cls_repeat*4+3], axis=-1), axis=-1, batch_dims=1)
            regression_score = self._proposal_score(x)
            regression_score = tf.gather(regression_score, cls_repeat[:, None], axis=-1, batch_dims=1)

            model_outputs.update({
                'regression_out': regression_out,
                'regression_score': regression_score,
            })

            if not training:
                box_out = _recover_boxes(bboxes.merge_dims(0,1), regression_out)
                ragged_box_out = tf.RaggedTensor.from_row_starts(box_out, bboxes.row_starts())
                model_outputs.update({'proposal_boxes': ragged_box_out})

            if labels is not None:
                padded_bboxes = bboxes.to_tensor(-1, shape=(None, 1600, 4))
                gt_bboxes = tf.cast(labels['bboxes'], tf.float32) / hwhw # ragged tensor
                padded_gt_bboxes = gt_bboxes.to_tensor(-1, shape=(None, 800, 4))
                matched_boxes, matched_indices, m_ious, _ = box_matching(padded_bboxes, padded_gt_bboxes)

                targets =  _compare_boxes(padded_bboxes, matched_boxes)

                unpadding_indices = tf.where(padded_bboxes[:,:,0] >=0)
                targets = tf.gather_nd(targets, unpadding_indices)
                m_ious = tf.gather_nd(m_ious[:,:,None], unpadding_indices)

                model_outputs.update({
                    'regression_targets': targets,
                    'm_ious': m_ious,
                })

        return model_outputs

    def _build_metrics(self):
        metrics = []
        metric_names = [
            'model_loss', 'ofs_loss', 'weight_loss', 'classification_loss', 'regression_score_loss', 'regression_loss',
            'max_iou', 'a50', 'a75', 'a95', 'r50', 'r75', 'r95',
        ]
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
        losses = {
            'model_loss': model_loss,
            'ofs_loss': ofs_loss,
            'weight_loss': weight_loss,
            'classification_loss': classification_loss,
        }

        if self.burn_in_is_done:
            gt_scores = tf.cast(model_out['m_ious'] > self._negative_iou, tf.int32)
            scores = model_out['regression_score']
            regression_score_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(gt_scores, scores, from_logits=True))
            pos_indices = tf.where(gt_scores > 0)
            regression_targets = tf.gather(model_out['regression_targets'], pos_indices)
            regression_out = tf.gather(model_out['regression_out'], pos_indices)
            regression_loss = tf.reduce_mean(tf.keras.losses.huber(regression_targets, regression_out, 0.1))
            model_loss += regression_score_loss + regression_loss
            losses.update({
                'model_loss': model_loss,
                'regression_score_loss': regression_score_loss,
                'regression_loss': regression_loss,
            })
        return losses

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
        # def mean_max_iou(proposal_img, cur_label):
        #     ious = proposal_iou(proposal_img, cur_label['mask_indices'])
        #     return tf.reduce_mean(tf.reduce_max(ious, axis=-1))
        #
        imgs, labels = inputs

        model_out = self(inputs, training=False)
        metrics = self._build_losses(labels, model_out)

        # preds = pred_labels(model_out['offsets'], model_out['weights'], eps=self._eps, min_weight=self._min_samples)
        # ious = tf.map_fn(
        #         lambda x: mean_max_iou(*x),
        #         (preds, labels),
        #         fn_output_signature=tf.TensorSpec(shape=(), dtype=tf.float32),
        # )
        # metrics.update({'max_iou': ious})

        if self.burn_in_is_done:
            h = tf.shape(imgs)[1]
            w = tf.shape(imgs)[2]
            box_out = model_out['proposal_boxes']
            padded_boxes = box_out.to_tensor(-1, shape=(None, 1600, 4))
            gt_boxes = tf.cast(labels['bboxes'] / [h,w,h,w], tf.float32)
            padded_gt_boxes = gt_boxes.to_tensor(-1, shape=(None, 800,4))
            _, matched_indices, m_ious, _ = box_matching(padded_boxes, padded_gt_boxes)

            score_out = model_out['regression_score'][:,0]
            score_out = tf.RaggedTensor.from_row_starts(score_out, box_out.row_starts())
            padded_score_out = score_out.to_tensor(-9999., shape=(None, 1600))
            m_ious = tf.where(padded_score_out>0, m_ious, -1.)

            n_matched = tf.math.count_nonzero(m_ious > 0, axis = -1)
            n_gt_masks = gt_boxes.row_lengths()

            count_unique =lambda x: tf.math.count_nonzero(tf.experimental.numpy.diff(tf.sort(x, axis=-1)))
            n_matched_50 = count_unique(tf.where(m_ious >= .5, matched_indices, -1))
            n_matched_75 = count_unique(tf.where(m_ious >= .75, matched_indices, -1))
            n_matched_95 = count_unique(tf.where(m_ious >= .95, matched_indices, -1))

            metrics.update({
                'a50': tf.cast(n_matched_50 / n_matched, tf.float32),
                'a75': tf.cast(n_matched_75 / n_matched, tf.float32),
                'a95': tf.cast(n_matched_95 / n_matched, tf.float32),
                'r50': tf.cast(n_matched_50 / n_gt_masks, tf.float32),
                'r75': tf.cast(n_matched_75 / n_gt_masks, tf.float32),
                'r95': tf.cast(n_matched_95 / n_gt_masks, tf.float32),
            })

        logs =  self.update_metrics(metrics)
        return logs
