import tensorflow as tf

class MaskLayer(tf.keras.layers.Layer):
    def __init__(self, n_cls=3, crop_layer=0, crop_size=32, min_score = 0.35, conv_channels=64, **kwargs):
        super(MaskLayer, self).__init__(**kwargs)
        self._config_dict = {
            'n_cls': n_cls,
            'crop_layer': crop_layer,
            'crop_size': crop_size,
            'min_score': min_score,
            'conv_channels': conv_channels,
        }

    def get_config(self):
        config = super(MaskLayer, self).get_config()
        config.update(self._config_dict)
        return config

    def build(self, input_shape, **kwargs):
        n_cls = self._config_dict['n_cls']
        conv_channels = self._config_dict['conv_channels']
        self._mask_block = [
            BatchConv2D(conv_channels, name='mask_conv1'),
            BatchConv2D(conv_channels, name='mask_conv1'),
        ]
        self._mask_out = tf.keras.layers.Conv2D(n_cls, 3, name='mask_out')

        super(MaskLayer,self).build(input_shape)

    def call(self, inputs, training=None):
        labels, model_out = inputs

        crop_size = self._config_dict['crop_size']
        crop_layer = self._config_dict['crop_layer']

        bboxes = model_out['bboxes_out']
        scores = model_out['bboxes_score_out']
        if labels is not None:
            cls = labels['class']
        else:
            cls = tf.argmax(model_out['cls'], axis=-1, output_type=tf.int32)
        cls_repeat = tf.repeat(cls, bboxes.row_lengths())

        features = model_out['decoder_out'][str(crop_layer)]
        feature_crops = crop_features(features, bboxes, crop_size)

        x = feature_crops
        for layer in self._mask_block:
            x = layer(x)
        masks = self._mask_out(x)
        masks = tf.gather(masks, cls_repeat[:, None], axis=-1, batch_dims=1)
        return masks

    def _build_losses(self, labels, model_out):
        _h,w,_ = labels['dist_map'].get_shape()
        crop_size = self._config_dict['crop_size']
        bboxes = model_out['bboxes_out']
        gt_bboxes = labels['bboxes']
        _, matched_ids, matched_ious, _ = ragged_box_matching(bboxes, gt_bboxes)
        gt_masks = crop_masks(labels['mask_indices'], bboxes, matched_ids, crop_size, h, w)
        pred_masks = model_out['masks']

        pos_indices = tf.where(matched_ious.values > self._config_dict['min_score'])
        gt_masks = tf.gather_nd(gt_masks.values, pos_indices)
        pred_masks = tf.gather_nd(pred_masks.values, pos_indices)
        mask_loss = tf.losses.binary_crossentropy(gt_masks, pred_masks, from_logits=True)
        return mask_loss
