import tensorflow as tf
import tensorflow_addons as tfa
from .common import *
from .unet import *
from ..ops import *

class ProposalLayer(tf.keras.layers.Layer):
    def __init__(self, n_cls=3, eps=1.0, min_samples=4.0, min_weight=0.01,
              crop_size=16, min_iou = 0.3, conv_channels=64,
              **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self._config_dict = {
            'n_cls': n_cls,
            'eps': eps,
            'min_samples': min_samples,
            'min_weight': min_weight,
            'crop_size': crop_size,
            'min_iou': min_iou,
            'conv_channels': conv_channels,
        }

    def get_config(self):
        base_config = super(ProposalLayer, self).get_config()
        return dict(list(base_config.items()) + list(self._config_dict.items()))

    def build(self, input_shape):
        n_cls = self._config_dict['n_cls']
        conv_channels = self._config_dict['conv_channels']
        proposal_block = []
        proposal_block.append(BatchConv2D(conv_channels, strides=2, name='proposal_conv1'))
        proposal_block.append(BatchConv2D(conv_channels * 2, strides=2, name='proposal_conv2'))
        proposal_block.append(tf.keras.layers.Flatten())
        proposal_block.append(tf.keras.layers.Dense(512, activation='relu', name='proposal_fc1'))
        proposal_block.append(tf.keras.layers.Dense(512, activation='relu', name='proposal_fc2'))
        self._proposal_block = proposal_block
        self._proposal_regression = tf.keras.layers.Dense(4 * n_cls, name='box_regression')
        self._proposal_score = tf.keras.layers.Dense(n_cls, name='score_out')

    def _guess_sizes(self, proposals, sizes, weights):
        def one_frame(inputs):
            p,s,w = inputs
            indicator = tf.one_hot(p, tf.reduce_max(p)+1, axis=0, dtype=tf.uint8)[..., None]
            indicator = tf.where(indicator > 0, w, 0.)
            return tf.reduce_sum(s * indicator, axis=(1,2)) / tf.reduce_sum(indicator, axis=(1,2))
        return tf.map_fn(
            one_frame,
            [proposals, sizes, weights],
            fn_output_signature=tf.RaggedTensorSpec((None, 2), tf.float32, 0),
        )

    def call(self, inputs, training = None, **kwargs):
        features, model_out, cls = inputs
        offsets = model_out['offsets']
        weights = model_out['weights']
        _, h, w, _= offsets.get_shape()
        eps = self._config_dict['eps']
        min_samples = self._config_dict['min_samples']
        min_weight = self._config_dict['min_weight']
        crop_size = self._config_dict['crop_size']

        proposals = pred_labels(offsets, weights, eps=eps, min_samples=min_samples, min_weight=min_weight)
        bboxes = tf.cast(bbox_of_proposals(proposals), tf.float32) / [h,w,h,w]  # ragged tensor
        if 'sizes' in model_out:
            coded_flatten_bboxes = box_encode(tf.cast(bboxes.values, tf.float32))
            sizes = self._guess_sizes(proposals, model_out['sizes'], weights) / [h, w]
            #new_bboxes = tf.stop_gradient(tf.concat([coded_flatten_bboxes[:,0:2], sizes.values], axis=-1))
            new_bboxes = tf.concat([coded_flatten_bboxes[:,0:2], sizes.values], axis=-1)
            new_bboxes = tf.stop_gradient(box_decode(new_bboxes))
            bboxes = tf.RaggedTensor.from_row_starts(new_bboxes, bboxes.row_starts())
        feature_crops = crop_features(features, bboxes, crop_size)
        cls_repeat = tf.repeat(cls, bboxes.row_lengths())

        x = feature_crops
        for layer in self._proposal_block:
            x = layer(x, training=training)
        regression_out = self._proposal_regression(x, training=training)
        regression_out = tf.gather(regression_out, tf.stack([cls_repeat*4, cls_repeat*4+1, cls_repeat*4+2, cls_repeat*4+3], axis=-1), axis=-1, batch_dims=1)
        regression_score = self._proposal_score(x, training=training)
        regression_score = tf.gather(regression_score, cls_repeat[:, None], axis=-1, batch_dims=1)
        regression_bboxes = recover_boxes(bboxes.values, regression_out)

        outputs = {
            'proposal_bboxes': bboxes, #ragged
            'regressions': regression_out,
            'regression_scores': regression_score[:,0],
            'regression_bboxes': regression_bboxes,
        }

        return outputs

    def build_losses(self, labels, model_out):
        _, oh, ow, _ = labels['dist_map'].get_shape()
        bboxes = model_out['proposal_bboxes']
        gt_bboxes = tf.cast(labels['bboxes'], tf.float32) / [oh, ow, oh, ow] # ragged tensor
        matched_bboxes, matched_indices, matched_ious, _ = ragged_box_matching(bboxes, gt_bboxes)

        gt_scores = matched_ious.values
        scores = model_out['regression_scores']
        regression_score_loss = tf.reduce_mean(tf.losses.huber(gt_scores, scores, 0.5))

        pos_indices = tf.where(gt_scores > self._config_dict['min_iou'])
        good_proposal_bboxes = tf.gather(bboxes.values, pos_indices)
        gt_good_bboxes = tf.gather(matched_bboxes.values, pos_indices)
        #pred_good_bboxes = tf.gather(model_out['regression_bboxes'], pos_indices)
        regression_targets = compare_boxes(good_proposal_bboxes, gt_good_bboxes)
        regression_results = tf.gather(model_out['regressions'], pos_indices)
        regression_loss = tf.reduce_mean(tf.losses.huber(regression_targets, regression_results, 0.5))
        if tf.is_nan(regression_loss):
            regression_loss = tf.constant(0)
        #regression_loss = tf.reduce_mean(tfa.losses.giou_loss(gt_good_bboxes, pred_good_bboxes, 'giou'))

        losses = {
            'proposal_score_loss': regression_score_loss,
            'proposal_loss': regression_loss,
        }

        return losses
