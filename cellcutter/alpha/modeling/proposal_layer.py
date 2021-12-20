import tensorflow as tf
from .common import *
from .unet import *
from ..ops import *

def giou_loss(y_true: TensorLike, y_pred: TensorLike, mode: str = "giou") -> tf.Tensor:
    """Implements the GIoU loss function.
    GIoU loss was first introduced in the
    [Generalized Intersection over Union:
    A Metric and A Loss for Bounding Box Regression]
    (https://giou.stanford.edu/GIoU.pdf).
    GIoU is an enhancement for models which use IoU in object detection.
    Args:
        y_true: true targets tensor. The coordinates of the each bounding
            box in boxes are encoded as [y_min, x_min, y_max, x_max].
        y_pred: predictions tensor. The coordinates of the each bounding
            box in boxes are encoded as [y_min, x_min, y_max, x_max].
        mode: one of ['giou', 'iou'], decided to calculate GIoU or IoU loss.
    Returns:
        GIoU loss float `Tensor`.
    """
    if mode not in ["giou", "iou"]:
        raise ValueError("Value of mode should be 'iou' or 'giou'")
    y_pred = tf.convert_to_tensor(y_pred)
    if not y_pred.dtype.is_floating:
        y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, y_pred.dtype)
    giou = tf.squeeze(_calculate_giou(y_pred, y_true, mode))

    return 1 - giou


def _calculate_giou(b1: TensorLike, b2: TensorLike, mode: str = "giou") -> tf.Tensor:
    """
    Args:
        b1: bounding box. The coordinates of the each bounding box in boxes are
            encoded as [y_min, x_min, y_max, x_max].
        b2: the other bounding box. The coordinates of the each bounding box
            in boxes are encoded as [y_min, x_min, y_max, x_max].
        mode: one of ['giou', 'iou'], decided to calculate GIoU or IoU loss.
    Returns:
        GIoU loss float `Tensor`.
    """
    zero = tf.convert_to_tensor(0.0, b1.dtype)
    b1_ymin, b1_xmin, b1_ymax, b1_xmax = tf.unstack(b1, 4, axis=-1)
    b2_ymin, b2_xmin, b2_ymax, b2_xmax = tf.unstack(b2, 4, axis=-1)
    b1_width = tf.maximum(zero, b1_xmax - b1_xmin)
    b1_height = tf.maximum(zero, b1_ymax - b1_ymin)
    b2_width = tf.maximum(zero, b2_xmax - b2_xmin)
    b2_height = tf.maximum(zero, b2_ymax - b2_ymin)
    b1_area = b1_width * b1_height
    b2_area = b2_width * b2_height

    intersect_ymin = tf.maximum(b1_ymin, b2_ymin)
    intersect_xmin = tf.maximum(b1_xmin, b2_xmin)
    intersect_ymax = tf.minimum(b1_ymax, b2_ymax)
    intersect_xmax = tf.minimum(b1_xmax, b2_xmax)
    intersect_width = tf.maximum(zero, intersect_xmax - intersect_xmin)
    intersect_height = tf.maximum(zero, intersect_ymax - intersect_ymin)
    intersect_area = intersect_width * intersect_height

    union_area = b1_area + b2_area - intersect_area
    iou = tf.math.divide_no_nan(intersect_area, union_area)
    if mode == "iou":
        return iou

    enclose_ymin = tf.minimum(b1_ymin, b2_ymin)
    enclose_xmin = tf.minimum(b1_xmin, b2_xmin)
    enclose_ymax = tf.maximum(b1_ymax, b2_ymax)
    enclose_xmax = tf.maximum(b1_xmax, b2_xmax)
    enclose_width = tf.maximum(zero, enclose_xmax - enclose_xmin)
    enclose_height = tf.maximum(zero, enclose_ymax - enclose_ymin)
    enclose_area = enclose_width * enclose_height
    giou = iou - tf.math.divide_no_nan((enclose_area - union_area), enclose_area)
    return giou

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
        regression_loss = tf.reduce_mean(giou_loss(gt_boxes, pred_boxes))
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
