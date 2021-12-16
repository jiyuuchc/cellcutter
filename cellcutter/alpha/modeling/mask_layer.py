import tensorflow as tf
from .common import *
from ..ops import *
import skimage.transform

# def cleanup(boxes, mis, scores, h=544, w=704):
#     def one_img(inputs):
#         box,mi,score = inputs
#         ious = ious_of_masks(mi, mi, h, w)
#         indices = tf.image.non_max_suppression_overlaps(ious, score, 2000, overlap_threshold=0.25)
#         ragged_mi = tf.RaggedTensor.from_value_rowids(mi[:,1:3], mi[:,0])
#         ragged_mi = tf.gather(ragged_mi, indices)
#         mi = tf.concat([ragged_mi.value_rowids()[:,None], ragged_mi.values], axis=-1)
#         score = tf.gather(score, indices)
#         box = tf.gather(box, indices)
#         return box, mi, score
#     return tf.map_fn(
#         one_img,
#         [boxes, mis, scores],
#         fn_output_signature=(
#             tf.RaggedTensorSpec((None,4), tf.float32, 0),
#             tf.RaggedTensorSpec((None,3), tf.int32, 0),
#             tf.RaggedTensorSpec((None,), tf.float32, 0),
#             ),
#     )

def decode_one_img(masks_one_img, bboxes_one_img, scores_one_img, threshold, h, w):
    bboxes_one_img = tf.ensure_shape(bboxes_one_img, (None,4))
    def decode_one_mask(inputs):
        mask, box = inputs
        r0 = box[0]
        r1 = box[2]
        c0 = box[1]
        c1 = box[3]
        if r1<r0 or c1<c0:
            return tf.zeros((0,2),tf.int32)
        # if box[2] == box[0] or box[3] == box[1]:
        #     return tf.zeros((h,w), tf.bool)
        # if box[2] > box[0]:
        #     r0 = box[0]
        #     r1 = box[2]
        # else:
        #     r0 = box[2]
        #     r1 = box[0]
        #     mask = mask[::-1,:]
        # if box[3] > box[1]:
        #     c0 = box[1]
        #     c1 = box[3]
        # else:
        #     c0 = box[3]
        #     c1 = box[1]
        #     mask = mask[:,::-1]
        mask = tf.image.resize(mask, [r1-r0+1, c1-c0+1])
        mi = tf.cast(tf.where(mask[:,:,0] >= .5), dtype=tf.int32) + [r0,c0]
        # indices = tf.where((mi[:,0]>=0) & (mi[:,0]<h) & (mi[:,1]>=0) & (mi[:,1]<w))
        # if tf.size(indices) == 0:
        #     return tf.zeros((h,w), tf.bool)
        # mi = tf.gather_nd(mi, indices)
        # return tf.scatter_nd(mi, tf.ones((tf.shape(mi)[0],), tf.bool), [h,w])
        return mi
    bboxes_one_img = tf.cast(tf.math.round(bboxes_one_img * [h, w, h, w]), tf.int32)
    mis = tf.map_fn(
        decode_one_mask,
        [tf.cast(masks_one_img, tf.float32), bboxes_one_img],
        fn_output_signature=tf.RaggedTensorSpec((None, 2), tf.int32, 0),
    )
    miv = mis.values
    mir = tf.cast(mis.value_rowids(), tf.int32)
    indices = tf.where((miv[:,0]>=0) & (miv[:,0]<h) & (miv[:,1]>=0) & (miv[:,1]<w))
    miv = tf.gather_nd(miv, indices)
    mir = tf.gather_nd(mir, indices)
    return tf.concat([mir[:,None], miv], axis=-1), scores_one_img
    # mis = tf.concat([mir[:,None], miv], axis=-1)
    # n_masks = tf.shape(scores_one_img)[0]
    # mask_stack = tf.scatter_nd(mis, tf.ones((tf.shape(mis)[0],), tf.bool), [n_masks, h, w])
    # mask_areas = tf.math.count_nonzero(mask_stack, axis=(1,2))
    #
    # def iou_one_row(inputs):
    #     one_mask, one_mask_area = inputs
    #     intersects = tf.math.count_nonzero(one_mask & mask_stack, axis=(1,2))
    #     union = one_mask_area + mask_areas - intersects
    #     return tf.cast(intersects, tf.float32) / (tf.cast(union, tf.float32) + 1.0e-7)
    # ious = tf.map_fn(
    #     iou_one_row,
    #     [mask_stack, mask_areas],
    #     fn_output_signature=tf.TensorSpec((None,), tf.float32),
    # )
    #
    # indices = tf.image.non_max_suppression_overlaps(ious, scores_one_img, 2000, overlap_threshold=threshold)
    # mask_stack = tf.gather(mask_stack, indices)
    # scores = tf.gather(scores_one_img, indices)
    # return tf.where(mask_stack), scores

class MaskLayer(tf.keras.layers.Layer):
    def __init__(self, n_cls=3, crop_layer=0, crop_size=32, n_convs=3, conv_channels=48, min_iou=0.5, min_score=0.2, min_overlap=0.2, **kwargs):
        super(MaskLayer, self).__init__(**kwargs)
        self._config_dict = {
            'n_cls': n_cls,
            'crop_layer': crop_layer,
            'crop_size': crop_size,
            'n_convs': n_convs,
            'min_score': min_score,
            'min_iou': min_iou,
            'min_overlap': min_overlap,
            'conv_channels': conv_channels,
        }

    def get_config(self):
        config = super(MaskLayer, self).get_config()
        config.update(self._config_dict)
        return config

    def build(self, input_shape, **kwargs):
        n_cls = self._config_dict['n_cls']
        conv_channels = self._config_dict['conv_channels']
        n_convs = self._config_dict['n_convs']

        self._mask_block = []
        for k in range(n_convs):
            self._mask_block.append(BatchConv2D(conv_channels, name=f'mask_conv{k+1}'))
        self._mask_out = tf.keras.layers.Conv2D(n_cls, 1, name='mask_out')

        super(MaskLayer,self).build(input_shape)

    def call(self, inputs, training=None):
        (decoder_out, bboxes, scores, cls), labels = inputs
        _, h, w, _ = decoder_out['0'].get_shape()
        crop_size = self._config_dict['crop_size']
        crop_layer = self._config_dict['crop_layer']

        # bboxes = model_out['bboxes_out']
        # scores = model_out['bboxes_score_out']
        if training:
            gt_bboxes = tf.cast(labels['bboxes'], tf.float32) / [h, w, h, w]
            _, mIds, mIous, _ = ragged_box_matching(bboxes, gt_bboxes)
            is_broken = tf.gather(labels['is_broken'], mIds, batch_dims=1)
            mIous = tf.where(is_broken, 0.0, mIous)
            indices = tf.where(mIous > self._config_dict['min_iou'])
            indices = tf.RaggedTensor.from_value_rowids(indices[:,1], indices[:,0])
            mIds = tf.gather(mIds, indices, batch_dims=1)
            bboxes = tf.gather(bboxes, indices, batch_dims=1)
            scores = tf.gather(scores, indices, batch_dims=1)
        else:
            indices = tf.where(scores > self._config_dict['min_score'])
            indices = tf.RaggedTensor.from_value_rowids(indices[:,1], indices[:,0])
            bboxes = tf.gather(bboxes, indices, batch_dims=1)
            scores = tf.gather(scores, indices, batch_dims=1)

        bboxes = tf.math.round(bboxes * [h, w, h, w])
        v = bboxes.values
        r = bboxes.row_lengths()
        v = tf.concat([v[:,:2], v[:,2:]-1.0], axis=-1) / [h-1, w-1, h-1, w-1]
        bboxes = tf.RaggedTensor.from_row_lengths(v, r)

        features = decoder_out[str(crop_layer)]
        feature_crops = crop_features(features, bboxes, crop_size)

        x = feature_crops
        for layer in self._mask_block:
            x = layer(x)
        masks = self._mask_out(x)

        # if labels is not None:
        #     cls = labels['class']
        # else:
        #     cls = tf.argmax(model_out['cls'], axis=-1, output_type=tf.int32)
        cls_repeat = tf.repeat(cls, bboxes.row_lengths())
        masks = tf.gather(masks, cls_repeat[:, None], axis=-1, batch_dims=1)
        masks = tf.RaggedTensor.from_row_starts(masks, bboxes.row_starts())

        if training:
            return {
                'masks': masks,
                'mask_bboxes': bboxes,
                'mask_mIds': mIds,
            }
        else:
            return {
                'masks': masks,
                'mask_bboxes': bboxes,
                'mask_scores': scores,
            }

    def _build_losses(self, labels, model_out):
        _, h, w, _ = labels['dist_map'].get_shape()
        crop_size = self._config_dict['crop_size']
        bboxes = model_out['mask_bboxes']
        matched_ids = model_out['mask_mIds']
        gt_masks = crop_masks(labels['mask_indices'], bboxes, matched_ids, crop_size, h, w)
        pred_masks = model_out['masks'].values
        mask_loss = tf.reduce_mean(tf.losses.binary_crossentropy(gt_masks, pred_masks, from_logits=True))
        return mask_loss

    def predict(self, inputs):
        decoder_out, _, _, _ = inputs
        _, h, w, _ = decoder_out['0'].get_shape()
        min_overlap = self._config_dict['min_overlap']
        layer_out = self((inputs, None), training=False)
        scores = layer_out['mask_scores']
        masks = tf.sigmoid(layer_out['masks'])
        boxes = layer_out['mask_bboxes']

        mask_indices, scores = tf.map_fn(
            lambda x: decode_one_img(x[0], x[1], x[2], min_overlap, h, w),
            [masks, boxes, scores],
            fn_output_signature=(tf.RaggedTensorSpec((None,3), tf.int32, 0), tf.RaggedTensorSpec((None,), tf.float32, 0)),
        )
        return boxes, mask_indices, scores
