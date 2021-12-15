import tensorflow as tf
import tensorflow_addons as tfa
from .common import *
from .unet import *
from ..ops import *

class ProposalLayer(tf.keras.layers.Layer):
    def __init__(self, n_cls=3, eps=1.0, min_weight=0.01,
              crop_layer=1, crop_size=16, min_iou = 0.35,
              conv_channels=64, fc_channels=512, use_dbscan=False,
              **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self._config_dict = {
            'n_cls': n_cls,
            'eps': eps,
            'min_weight': min_weight,
            'crop_layer': crop_layer,
            'crop_size': crop_size,
            'min_iou': min_iou,
            'conv_channels': conv_channels,
            'fc_channels': fc_channels,
            'use_dbscan': use_dbscan,
        }
        self._pg = ProposalGenerator(n_cls, eps, min_weight, use_dbscan=use_dbscan)

    def get_config(self):
        config = super(ProposalLayer, self).get_config()
        config.update(self._config_dict)
        return config

    def build(self, input_shape, **kwargs):
        n_cls = self._config_dict['n_cls']
        conv_channels = self._config_dict['conv_channels']
        fc_channels = self._config_dict['fc_channels']
        proposal_block = []
        proposal_block.append(BatchConv2D(conv_channels, strides=2, name='proposal_conv1'))
        proposal_block.append(BatchConv2D(conv_channels * 2, strides=2, name='proposal_conv2'))
        proposal_block.append(tf.keras.layers.Flatten())
        proposal_block.append(tf.keras.layers.Dense(fc_channels, activation='relu', name='proposal_fc1'))
        proposal_block.append(tf.keras.layers.Dense(fc_channels, activation='relu', name='proposal_fc2'))
        self._proposal_block = proposal_block
        self._proposal_regression = tf.keras.layers.Dense(4 * n_cls, name='box_regression')
        self._proposal_score = tf.keras.layers.Dense(n_cls, name='score_out')

        super(ProposalLayer,self).build(input_shape)

    def call(self, inputs, training = None, **kwargs):
        labels, model_out = inputs

        offsets = model_out['offsets']
        weights = model_out['weights']
        _, h, w, _= offsets.get_shape()

        bboxes, scores = self._pg(labels, model_out)
        outputs = {
            'proposal_bboxes': bboxes,
            'proposal_scores': scores,
        }
        tf.debugging.assert_positive(bboxes.row_lengths(), 'Proposal box list is empty')

        crop_size = self._config_dict['crop_size']
        crop_layer = self._config_dict['crop_layer']
        # bboxes = outputs['proposal_bboxes']  # ragged tensor
        features = model_out['decoder_out'][str(crop_layer)]
        feature_crops = crop_features(features, bboxes, crop_size)
        if labels is not None:
            cls = labels['class']
        else:
            cls = tf.argmax(model_out['cls'], axis=-1, output_type=tf.int32)
        cls_repeat = tf.repeat(cls, bboxes.row_lengths())

        x = feature_crops
        for layer in self._proposal_block:
            x = layer(x, training=training)
        regression_out = self._proposal_regression(x, training=training)
        regression_out = tf.gather(regression_out, tf.stack([cls_repeat*4, cls_repeat*4+1, cls_repeat*4+2, cls_repeat*4+3], axis=-1), axis=-1, batch_dims=1)
        regression_score = self._proposal_score(x, training=training)
        regression_score = tf.gather(regression_score, cls_repeat[:, None], axis=-1, batch_dims=1)
        regression_bboxes = recover_boxes(bboxes.values, regression_out)

        outputs.update({
            'regressions': regression_out,
            'regression_scores': regression_score[:,0],
            'regression_bboxes': regression_bboxes,
        })

        return outputs

    def build_losses(self, labels, model_out):
        _,oh,ow,_ = labels['dist_map'].get_shape()
        gt_bboxes = tf.cast(labels['bboxes'], tf.float32) / [oh, ow, oh, ow] # ragged tensor
        bboxes = model_out['proposal_bboxes']
        matched_bboxes, matched_indices, matched_ious, _ = ragged_box_matching(bboxes, gt_bboxes)
        pos_indices = tf.where(matched_ious.values > self._config_dict['min_iou'])

        pred_boxes = tf.gather(model_out['regression_bboxes'], pos_indices)
        gt_boxes = tf.gather(matched_bboxes.values, pos_indices)
        regression_loss = tf.reduce_mean(tfa.losses.giou_loss(gt_boxes, pred_boxes))
        # good_proposal_bboxes = tf.gather(bboxes.values, pos_indices)
        # gt_good_bboxes = tf.gather(matched_bboxes.values, pos_indices)
        # regression_results = tf.gather(model_out['regressions'], pos_indices)
        # regression_targets = compare_boxes(good_proposal_bboxes, gt_good_bboxes)
        # regression_loss = tf.reduce_mean(tf.losses.huber(regression_targets, regression_results, 0.5))

        gt_scores = box_ious(tf.stop_gradient(model_out['regression_bboxes']), tf.stop_gradient(matched_bboxes.values))
        tf.debugging.assert_all_finite(gt_scores, 'ious computation returned unexpected Nan or Inf')
        scores = model_out['regression_scores']
        tf.debugging.assert_positive(tf.size(scores), 'Proposal box list is empty')
        regression_score_loss = tf.reduce_mean(tf.losses.huber(gt_scores, scores, 0.5))

        losses = {
            'proposal_score_loss': regression_score_loss,
            'proposal_loss': regression_loss,
        }

        return losses
