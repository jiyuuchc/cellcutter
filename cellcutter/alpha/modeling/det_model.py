''' UNet based cell-detection model '''

import tensorflow as tf
from ..ops import *
from .common import *
from .unet import *
from .losses import iou_loss

class DetModel(tf.keras.Model):
    def __init__(
        self,
        encoder,
        decoder,
        proposal_layer,
        n_cls = 3,
        regression_conv_channels = 128,
        classification_fc_channels =128,
        regression_layer = 0,
        classification_layer = 4,
    ):
        super(DetModel,self).__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._proposal_layer = proposal_layer
        self._config_dict = {
            'encoder': encoder,
            'decoder': decoder,
            'proposal_layer': proposal_layer,
            'n_cls': n_cls,
            'regression_conv_channels': regression_conv_channels,
            'classification_fc_channels': classification_fc_channels,
            'regression_layer': regression_layer,
            'classification_layer': classification_layer,
        }

        self._metrics = self._build_metrics()

        self.start_full_training(False)

        n_ch = self._config_dict['regression_conv_channels']
        self._conv_block = [
            BatchConv2D(n_ch, name='regression_conv1'),
            BatchConv2D(n_ch, name='regression_conv2'),
        ]
        self._size_conv_block = [
            BatchConv2D(n_ch, name='regression_conv1'),
            BatchConv2D(n_ch, name='regression_conv2'),
        ]
        self._ofs_regression = tf.keras.layers.Conv2D(2 * n_cls, 1, padding='same', name='ofs_regression')
        self._weight_layer = tf.keras.layers.Conv2D(n_cls, 1, padding='same', name='weights')
        self._size_regression = tf.keras.layers.Conv2D(2 * n_cls, 1, padding='same', name='size_regression')

        self._classification_block = [
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(classification_fc_channels, activation='relu'),
            tf.keras.layers.Dropout(.2),
            tf.keras.layers.Dense(classification_fc_channels, activation='relu'),
            tf.keras.layers.Dropout(.2),
            tf.keras.layers.Dense(n_cls, activation='softmax'),
        ]

        # self._segmentation_conv_block = [
        #     BatchConv2D(self._segmentation_conv_channels, name='seg_conv1'),
        #     BatchConv2D(self._segmentation_conv_channels, name='seg_conv2'),
        # ]
        # self._segmentation_out = tf.keras.layers.Conv2D(self._n_classes, 3, padding='same')
        # self._segmentation_score_block = [
        #     layers.AveragePooling2D((4,4), name='score_pooling'),
        #     layers.Flatten(name='score_flatten'),
        #     layers.Dense(self._segmentation_fc_channels, activation='relu', name='score_dense1'),
        #     layers.Dense(self._segmentation_fc_channels, activation='relu', name='score_dense2'),
        #     layers.Dense(self._n_classes, name='score_out'),
        # ]

    def get_config(self):
        return self._config_dict

    def call(self, inputs, compute_mask=False, training = None):
        image, labels = inputs
        _,h,w,_ = image.get_shape()
        regression_layer = self._config_dict['regression_layer']
        classification_layer = self._config_dict['classification_layer']

        features = self._encoder(image, training=training)
        outputs = self._decoder(features, training=training)

        #x = outputs[str(regression_layer)]
        x = outputs[str(regression_layer)]
        for layer in self._conv_block:
            x = layer(x, training=training)
        offsets = self._ofs_regression(x)
        weights = self._weight_layer(x)

        x = outputs[str(regression_layer)]
        for layer in self._size_conv_block:
            x = layer(x, training=training)
        sizes = self._size_regression(x)

        x=outputs[str(classification_layer)]
        for layer in self._classification_block:
            x = layer(x, training=training)
        pred_cls = x

        if training and labels is not None:
            cls = labels['class']
        else:
            cls = tf.argmax(pred_cls, axis=-1, output_type=tf.int32)

        offsets = tf.gather(offsets, tf.stack([cls*2, cls*2+1], axis=-1), axis=-1, batch_dims=1)
        sizes = tf.gather(sizes, tf.stack([cls*2, cls*2+1], axis=-1), axis=-1, batch_dims=1)
        weights = tf.gather(weights, cls[:,None], axis=-1, batch_dims=1)

        model_outputs = {
            'offsets': offsets,
            'weights': weights,
            'sizes': sizes,
            'cls': pred_cls,
        }
        if not compute_mask:
            return model_outputs

        proposal_out = self._proposal_layer(
            (outputs[str(regression_layer)], model_outputs, cls),
            training=training,
        )

        model_outputs.update(proposal_out)

        # scale_factor = tf.constant([h, w, h, w], tf.float32) / 2**regression_layer
        # proposals = pred_labels(offsets, weights)
        # proposal_bboxes = tf.cast(bbox_of_proposals(proposals), tf.float32) / scale_factor
        # cls_repeat = tf.repeat(cls, proposal_bboxes.row_lengths())
        # x = crop_features(outputs['0'], proposal_bboxes, self._crop_size)
        # for layer in self._segmentation_conv_block:
        #     x = layer(x)
        # segmentation_out = self._segmentation_out(x)
        # segmentation_out = tf.gather(segmentation_out, cls_repeat[:,None], axis=-1, batch_dims=1)
        # for layer in self._segmentation_score_block:
        #     x = layer(x)
        # segmentation_score = tf.gather(x, cls_repeat[:,None], axis=-1, batch_dims=1)
        #
        # model_outputs.update({
        #     'segmentations': segmentation_out,
        #     'segmentation_score': segmentation_score,
        #     'proposal_bboxes': proposal_bboxes,
        # })
        return model_outputs

    def _build_metrics(self):
        metrics = []
        metric_names = [
            'model_loss', 'ofs_loss', 'weight_loss', 'classification_loss', 'size_loss',
            'proposal_score_loss', 'proposal_loss'
            #'segmentation_loss', 'score_loss',
            #'max_iou', 'a50', 'a75', 'a95', 'r50', 'r75', 'r95',
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

    def _build_losses(self, inputs, model_out, compute_mask=False):
        images, labels = inputs
        regression_layer = self._config_dict['regression_layer']
        ss = 2 ** regression_layer
        gt_offsets = tf.nn.avg_pool2d(labels['dist_map'], ss, ss, 'SAME') / ss
        gt_sizes = tf.nn.avg_pool2d(labels['size_map'], ss, ss, 'SAME') / ss
        gt_weights = tf.nn.avg_pool2d(labels['weights'], ss, ss, 'SAME')
        # gt_offsets = labels['dist_map']
        # gt_sizes = labels['size_map']
        # gt_weights = labels['weights']
        gt_cls = tf.expand_dims(labels['class'], -1)

        pred_cls = model_out['cls']
        offsets = model_out['offsets']
        sizes = model_out['sizes']
        weights = model_out['weights']

        w = tf.cast(gt_weights[...,0] > 0.1, tf.float32)
        ofs_loss = tf.losses.huber(gt_offsets, offsets, 200)
        ofs_loss = tf.reduce_mean(tf.reduce_sum(ofs_loss * w, axis=(1,2))/tf.reduce_sum(w, axis=(1,2)))*ss*ss

        size_loss = tf.losses.huber(gt_sizes, sizes, 200)
        w = tf.cast(gt_weights[...,0] > 0.5, tf.float32)
        size_loss = 0.25 * ss * ss * tf.reduce_mean(tf.reduce_sum(size_loss * w, axis=(1,2)) / tf.reduce_sum(w, axis=(1,2)))

        weight_loss = tf.reduce_mean(tf.losses.binary_crossentropy(gt_weights, weights, from_logits=True))
        classification_loss = tf.reduce_mean(tf.losses.categorical_crossentropy(tf.one_hot(gt_cls[:,0], self._config_dict['n_cls']), pred_cls))

        model_loss = ofs_loss + weight_loss + size_loss

        losses = {
            'ofs_loss': ofs_loss,
            'size_loss': size_loss,
            'weight_loss': weight_loss,
            'classification_loss': classification_loss,
        }
        if compute_mask:
            proposal_losses = self._proposal_layer.build_losses(labels, model_out)
            model_loss += proposal_losses['proposal_score_loss']
            model_loss += proposal_losses['proposal_loss']
            losses.update(proposal_losses)
        # _,h,w,_ = images.get_shape()
        # bboxes = model_out['proposal_bboxes']
        # gt_bboxes = tf.cast(labels['bboxes'] / [h,w,h,w], tf.float32)
        # matched_bboxes, matched_indices, matched_ious, _ = ragged_box_matching(bboxes, gt_bboxes)
        # gt_masks = crop_masks(labels['mask_indices'], bboxes, matched_indices,  self._crop_size, h, w)
        # pos_indices = tf.where(matched_ious.merge_dims(0,1) > self._positive_iou)
        # masks = tf.gather_nd(model_out['segmentations'], pos_indices)
        # gt_masks = tf.gather_nd(gt_masks, pos_indices)
        # segmentation_loss = tf.reduce_mean(iou_loss(gt_masks, masks, from_logits=True))
        #
        # scores = model_out['segmentation_score']
        # score_loss = tf.reduce_mean(tf.losses.huber(matched_ious.merge_dims(0,1), scores, 0.5))
        #
        # model_loss += segmentation_loss + score_loss
        #
        # losses.update({
        #     'model_loss': model_loss,
        #     'segmentation_loss': segmentation_loss,
        #     'score_loss': score_loss,
        # })
        losses.update({
            'model_loss': model_loss,
        })
        return losses

    def _train_step(self, inputs, compute_mask=False):
        with tf.GradientTape() as tape:
            model_out = self(inputs, training=True, compute_mask=compute_mask)
            losses = self._build_losses(inputs, model_out, compute_mask=compute_mask)
        grads = tape.gradient(losses['model_loss'], self.trainable_variables)
        self.optimizer.apply_gradients((g, v) for (g, v) in zip(grads, self.trainable_variables) if g is not None)

        logs = self.update_metrics(losses)
        return logs

    def _test_step(self, inputs, compute_mask=False):
        imgs, labels = inputs
        model_out = self(inputs, training=False, compute_mask=compute_mask)
        metrics = self._build_losses(inputs, model_out, compute_mask=compute_mask)

        # gt_bboxes = tf.cast(labels['bboxes'], tf.float32)
        # n_gt_masks = tf.reduce_sum(gt_bboxes.row_lengths())
        # proposals = pred_labels(model_out['offsets'], model_out['weights'], eps=1.0, min_samples=4.)
        # bboxes = tf.cast(bbox_of_proposals(proposals), tf.float32) # ragged tensor
        # _, matched_indices, matched_ious, _ = ragged_box_matching(bboxes, gt_bboxes)
        #
        # metrics.update({
        #     'a50': tf.math.count_nonzero(matched_ious.merge_dims(0,1)>0.5) / n_gt_masks,
        #     'a75': tf.math.count_nonzero(matched_ious.merge_dims(0,1)>0.75) / n_gt_masks,
        #     'a95': tf.math.count_nonzero(matched_ious.merge_dims(0,1)>0.95) / n_gt_masks,
        # })

        logs =  self.update_metrics(metrics)
        return logs

    def start_full_training(self, full=True):
        self.train_step = lambda inputs: self._train_step(inputs, compute_mask=full)
        self.make_train_function(force=True)
        self.test_step = lambda inputs: self._test_step(inputs, compute_mask=full)
        self.make_test_function(force=True)
