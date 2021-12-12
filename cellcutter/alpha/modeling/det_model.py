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
        mask_layer = None,
        n_cls = 3,
        regression_conv_channels = 128,
        classification_fc_channels =128,
        regression_layer = 0,
        classification_layer = 4,
        score_threshold = 0.05,
        iou_threshold = 0.5,
        with_mask = False,
    ):
        super(DetModel,self).__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._proposal_layer = proposal_layer
        self._mask_layer = mask_layer
        self._config_dict = {
            'encoder': encoder,
            'decoder': decoder,
            'proposal_layer': proposal_layer,
            'mask_layer': mask_layer,
            'n_cls': n_cls,
            'regression_conv_channels': regression_conv_channels,
            'classification_fc_channels': classification_fc_channels,
            'regression_layer': regression_layer,
            'classification_layer': classification_layer,
            'score_threshold': score_threshold,
            'iou_threshold': iou_threshold,
            'with_mask': with_mask,
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
        self._ofs_regression = tf.keras.layers.Conv2D(2 * n_cls, 1, padding='same', name='ofs_out')
        self._weight_layer = tf.keras.layers.Conv2D(n_cls, 1, padding='same', name='weights_out')
        self._size_regression = tf.keras.layers.Conv2D(2 * n_cls, 1, padding='same', name='size_out')

        self._classification_block = [
            tf.keras.layers.Conv2D(classification_fc_channels, 3, strides=3, activation='relu', name='cls_conv1'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(classification_fc_channels, activation='relu', name='cls_fc1'),
            tf.keras.layers.Dense(classification_fc_channels, activation='relu', name='cls_fc2'),
            tf.keras.layers.Dense(n_cls, activation='softmax', name='cls_out'),
        ]

    def get_config(self):
        return self._config_dict

    def call(self, inputs, compute_mask=False, training = None):
        image, labels = inputs
        _,h,w,_ = image.get_shape()
        regression_layer = self._config_dict['regression_layer']
        classification_layer = self._config_dict['classification_layer']

        features = self._encoder(image, training=training)
        outputs = self._decoder(features, training=training)

        model_outputs = {
            'encoder_out': features,
            'decoder_out': outputs,
        }

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

        x=features[str(classification_layer)]
        for layer in self._classification_block:
            x = layer(x, training=training)
        pred_cls = x

        if labels is not None:
            cls = labels['class']
        else:
            cls = tf.argmax(pred_cls, axis=-1, output_type=tf.int32)

        offsets = tf.gather(offsets, tf.stack([cls*2, cls*2+1], axis=-1), axis=-1, batch_dims=1)
        sizes = tf.gather(sizes, tf.stack([cls*2, cls*2+1], axis=-1), axis=-1, batch_dims=1)
        weights = tf.gather(weights, cls[:,None], axis=-1, batch_dims=1)

        model_outputs.update({
            'offsets': offsets,
            'weights': weights,
            'sizes': sizes,
            'cls': pred_cls,
        })
        if not compute_mask:
            return model_outputs

        proposal_out = self._proposal_layer((labels, model_outputs), training=training,)
        model_outputs.update(proposal_out)

        score_threshold = self._config_dict['score_threshold']
        iou_threshold = self._config_dict['iou_threshold']
        proposal_bboxes = model_outputs['proposal_bboxes']
        rowlengths = proposal_bboxes.row_lengths()
        bboxes = tf.RaggedTensor.from_row_lengths(model_outputs['regression_bboxes'], rowlengths)
        scores = tf.RaggedTensor.from_row_lengths(model_outputs['regression_scores'], rowlengths)
        def cleanup(inputs):
            indices = tf.image.non_max_suppression(
                  *inputs,
                  max_output_size=2000,
                  score_threshold=score_threshold,
                  iou_threshold=iou_threshold,
            )
            sample_boxes = tf.gather(inputs[0], indices)
            sample_scores = tf.gather(inputs[1], indices)
            return sample_boxes, sample_scores
        bboxes,scores = tf.map_fn(
            cleanup,
            [bboxes, scores],
            fn_output_signature = (tf.RaggedTensorSpec((None,4),tf.float32,0), tf.RaggedTensorSpec((None,),tf.float32,0)),
        )
        model_outputs.update({
            'bboxes_out': bboxes,
            'bboxes_score_out': scores,
        })

        if self._config_dict['with_mask']:
            masks = self._mask_layer((labels, model_outputs), training=training)
            model_out.update({
                'masks': masks,
            })
        return model_outputs

    def _build_metrics(self):
        metrics = []
        metric_names = [
            'model_loss', 'ofs_loss', 'weight_loss', 'classification_loss', 'size_loss',
            'proposal_score_loss', 'proposal_loss','mask_loss',
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

        gt_cls = labels['class']
        pred_cls = model_out['cls']

        gt_offsets = tf.nn.avg_pool2d(labels['dist_map'], ss, ss, 'SAME') / ss
        gt_sizes = tf.nn.avg_pool2d(labels['size_map'], ss, ss, 'SAME') / ss
        gt_weights = tf.nn.avg_pool2d(labels['weights'], ss, ss, 'SAME')

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
        classification_loss = tf.reduce_mean(tf.losses.categorical_crossentropy(tf.one_hot(gt_cls, self._config_dict['n_cls']), pred_cls))

        model_loss = ofs_loss + weight_loss + size_loss + classification_loss
        #model_loss = ofs_loss + weight_loss + size_loss

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
            if self._config_dict['with_mask']:
                mask_loss = self._mask_layer._build_losses(labels, model_out)
                model_loss += mask_loss
                losses.update({
                    'mask_loss': mask_loss,
                })
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
        logs =  self.update_metrics(metrics)
        return logs

    def start_full_training(self, full=True):
        self.train_step = lambda inputs: self._train_step(inputs, compute_mask=full)
        self.make_train_function(force=True)
        self.test_step = lambda inputs: self._test_step(inputs, compute_mask=full)
        self.make_test_function(force=True)
