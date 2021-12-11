# Copied from official-vision package


# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Box matcher implementation."""


import tensorflow as tf


class BoxMatcher:
  """Matcher based on highest value.

  This class computes matches from a similarity matrix. Each column is matched
  to a single row.

  To support object detection target assignment this class enables setting both
  positive_threshold (upper threshold) and negative_threshold (lower thresholds)
  defining three categories of similarity which define whether examples are
  positive, negative, or ignored, for example:
  (1) thresholds=[negative_threshold, positive_threshold], and
      indicators=[negative_value, ignore_value, positive_value]: The similarity
      metrics below negative_threshold will be assigned with negative_value,
      the metrics between negative_threshold and positive_threshold will be
      assigned ignore_value, and the metrics above positive_threshold will be
      assigned positive_value.
  (2) thresholds=[negative_threshold, positive_threshold], and
      indicators=[ignore_value, negative_value, positive_value]: The similarity
      metric below negative_threshold will be assigned with ignore_value,
      the metrics between negative_threshold and positive_threshold will be
      assigned negative_value, and the metrics above positive_threshold will be
      assigned positive_value.
  """

  def __init__(self, thresholds, indicators, force_match_for_each_col=False):
    """Construct BoxMatcher.

    Args:
      thresholds: A list of thresholds to classify boxes into
        different buckets. The list needs to be sorted, and will be prepended
        with -Inf and appended with +Inf.
      indicators: A list of values to assign for each bucket. len(`indicators`)
        must equal to len(`thresholds`) + 1.
      force_match_for_each_col: If True, ensures that each column is matched to
        at least one row (which is not guaranteed otherwise if the
        positive_threshold is high). Defaults to False. If True, all force
        matched row will be assigned to `indicators[-1]`.

    Raises:
      ValueError: If `threshold` not sorted,
        or len(indicators) != len(threshold) + 1
    """
    if not all([lo <= hi for (lo, hi) in zip(thresholds[:-1], thresholds[1:])]):
      raise ValueError('`threshold` must be sorted, got {}'.format(thresholds))
    self.indicators = indicators
    if len(indicators) != len(thresholds) + 1:
      raise ValueError('len(`indicators`) must be len(`thresholds`) + 1, got '
                       'indicators {}, thresholds {}'.format(
                           indicators, thresholds))
    thresholds = thresholds[:]
    thresholds.insert(0, -float('inf'))
    thresholds.append(float('inf'))
    self.thresholds = thresholds
    self._force_match_for_each_col = force_match_for_each_col

  def __call__(self, similarity_matrix):
    """Tries to match each column of the similarity matrix to a row.

    Args:
      similarity_matrix: A float tensor of shape [N, M] representing any
        similarity metric.

    Returns:
      A integer tensor of shape [N] with corresponding match indices for each
      of M columns, for positive match, the match result will be the
      corresponding row index, for negative match, the match will be
      `negative_value`, for ignored match, the match result will be
      `ignore_value`.
    """
    squeeze_result = False
    if len(similarity_matrix.shape) == 2:
      squeeze_result = True
      similarity_matrix = tf.expand_dims(similarity_matrix, axis=0)

    static_shape = similarity_matrix.shape.as_list()
    num_rows = static_shape[1] or tf.shape(similarity_matrix)[1]
    batch_size = static_shape[0] or tf.shape(similarity_matrix)[0]

    def _match_when_rows_are_empty():
      """Performs matching when the rows of similarity matrix are empty.

      When the rows are empty, all detections are false positives. So we return
      a tensor of -1's to indicate that the columns do not match to any rows.

      Returns:
        matches:  int32 tensor indicating the row each column matches to.
      """
      with tf.name_scope('empty_gt_boxes'):
        matches = tf.zeros([batch_size, num_rows], dtype=tf.int32)
        match_labels = -tf.ones([batch_size, num_rows], dtype=tf.int32)
        return matches, match_labels

    def _match_when_rows_are_non_empty():
      """Performs matching when the rows of similarity matrix are non empty.

      Returns:
        matches:  int32 tensor indicating the row each column matches to.
      """
      # Matches for each column
      with tf.name_scope('non_empty_gt_boxes'):
        matches = tf.argmax(similarity_matrix, axis=-1, output_type=tf.int32)

        # Get logical indices of ignored and unmatched columns as tf.int64
        matched_vals = tf.reduce_max(similarity_matrix, axis=-1)
        matched_indicators = tf.zeros([batch_size, num_rows], tf.int32)

        match_dtype = matched_vals.dtype
        for (ind, low, high) in zip(self.indicators, self.thresholds[:-1],
                                    self.thresholds[1:]):
          low_threshold = tf.cast(low, match_dtype)
          high_threshold = tf.cast(high, match_dtype)
          mask = tf.logical_and(
              tf.greater_equal(matched_vals, low_threshold),
              tf.less(matched_vals, high_threshold))
          matched_indicators = self._set_values_using_indicator(
              matched_indicators, mask, ind)

        if self._force_match_for_each_col:
          # [batch_size, M], for each col (groundtruth_box), find the best
          # matching row (anchor).
          force_match_column_ids = tf.argmax(
              input=similarity_matrix, axis=1, output_type=tf.int32)
          # [batch_size, M, N]
          force_match_column_indicators = tf.one_hot(
              force_match_column_ids, depth=num_rows)
          # [batch_size, N], for each row (anchor), find the largest column
          # index for groundtruth box
          force_match_row_ids = tf.argmax(
              input=force_match_column_indicators, axis=1, output_type=tf.int32)
          # [batch_size, N]
          force_match_column_mask = tf.cast(
              tf.reduce_max(force_match_column_indicators, axis=1),
              tf.bool)
          # [batch_size, N]
          final_matches = tf.where(force_match_column_mask, force_match_row_ids,
                                   matches)
          final_matched_indicators = tf.where(
              force_match_column_mask, self.indicators[-1] *
              tf.ones([batch_size, num_rows], dtype=tf.int32),
              matched_indicators)
          return final_matches, final_matched_indicators
        else:
          return matches, matched_indicators

    num_gt_boxes = similarity_matrix.shape.as_list()[-1] or tf.shape(
        similarity_matrix)[-1]
    result_match, result_matched_indicators = tf.cond(
        pred=tf.greater(num_gt_boxes, 0),
        true_fn=_match_when_rows_are_non_empty,
        false_fn=_match_when_rows_are_empty)

    if squeeze_result:
      result_match = tf.squeeze(result_match, axis=0)
      result_matched_indicators = tf.squeeze(result_matched_indicators, axis=0)

    return result_match, result_matched_indicators

  def _set_values_using_indicator(self, x, indicator, val):
    """Set the indicated fields of x to val.

    Args:
      x: tensor.
      indicator: boolean with same shape as x.
      val: scalar with value to set.

    Returns:
      modified tensor.
    """
    indicator = tf.cast(indicator, x.dtype)
    return tf.add(tf.multiply(x, 1 - indicator), val * indicator)

def box_matching(boxes, gt_boxes):
  """Match boxes to groundtruth boxes.

  Given the proposal boxes and the groundtruth boxes and classes, perform the
  groundtruth matching by taking the argmax of the IoU between boxes and
  groundtruth boxes.

  Args:
    boxes: a tensor of shape of [batch_size, N, 4] representing the box
      coordiantes to be matched to groundtruth boxes.
    gt_boxes: a tensor of shape of [batch_size, MAX_INSTANCES, 4] representing
      the groundtruth box coordinates. It is padded with -1s to indicate the
      invalid boxes.

  Returns:
    matched_gt_boxes: a tensor of shape of [batch_size, N, 4], representing
      the matched groundtruth box coordinates for each input box. If the box
      does not overlap with any groundtruth boxes, the matched boxes of it
      will be set to all 0s.
    matched_gt_indices: a tensor of shape of [batch_size, N], representing
      the indices of the matched groundtruth boxes in the original gt_boxes
      tensor. If the box does not overlap with any groundtruth boxes, the
      index of the matched groundtruth will be set to -1.
    matched_iou: a tensor of shape of [batch_size, N], representing the IoU
      between the box and its matched groundtruth box. The matched IoU is the
      maximum IoU of the box and all the groundtruth boxes.
    iou: a tensor of shape of [batch_size, N, K], representing the IoU matrix
      between boxes and the groundtruth boxes. The IoU between a box and the
      invalid groundtruth boxes whose coordinates are [-1, -1, -1, -1] is -1.
  """
  # Compute IoU between boxes and gt_boxes.
  # iou <- [batch_size, N, K]
  iou = bbox_overlap(boxes, gt_boxes)

  # max_iou <- [batch_size, N]
  # 0.0 -> no match to gt, or -1.0 match to no gt
  matched_iou = tf.reduce_max(iou, axis=-1)

  # background_box_mask <- bool, [batch_size, N]
  background_box_mask = tf.less_equal(matched_iou, 0.0)

  argmax_iou_indices = tf.argmax(iou, axis=-1, output_type=tf.int32)

  argmax_iou_indices_shape = tf.shape(argmax_iou_indices)
  batch_indices = (
      tf.expand_dims(tf.range(argmax_iou_indices_shape[0]), axis=-1) *
      tf.ones([1, argmax_iou_indices_shape[-1]], dtype=tf.int32))
  gather_nd_indices = tf.stack([batch_indices, argmax_iou_indices], axis=-1)

  matched_gt_boxes = tf.gather_nd(gt_boxes, gather_nd_indices)
  matched_gt_boxes = tf.where(
      tf.tile(tf.expand_dims(background_box_mask, axis=-1), [1, 1, 4]),
      tf.zeros_like(matched_gt_boxes, dtype=matched_gt_boxes.dtype),
      matched_gt_boxes)

  matched_gt_indices = tf.where(background_box_mask,
                                -tf.ones_like(argmax_iou_indices),
                                argmax_iou_indices)

  return (matched_gt_boxes, matched_gt_indices, matched_iou,
          iou)

def bbox_overlap(boxes, gt_boxes):
  """Calculates the overlap between proposal and ground truth boxes.

  Some `gt_boxes` may have been padded.  The returned `iou` tensor for these
  boxes will be -1.

  Args:
    boxes: a tensor with a shape of [batch_size, N, 4]. N is the number of
      proposals before groundtruth assignment (e.g., rpn_post_nms_topn). The
      last dimension is the pixel coordinates in [ymin, xmin, ymax, xmax] form.
    gt_boxes: a tensor with a shape of [batch_size, MAX_NUM_INSTANCES, 4]. This
      tensor might have paddings with a negative value.

  Returns:
    iou: a tensor with as a shape of [batch_size, N, MAX_NUM_INSTANCES].
  """
  with tf.name_scope('bbox_overlap'):
    bb_y_min, bb_x_min, bb_y_max, bb_x_max = tf.split(
        value=boxes, num_or_size_splits=4, axis=2)
    gt_y_min, gt_x_min, gt_y_max, gt_x_max = tf.split(
        value=gt_boxes, num_or_size_splits=4, axis=2)

    # Calculates the intersection area.
    i_xmin = tf.math.maximum(bb_x_min, tf.transpose(gt_x_min, [0, 2, 1]))
    i_xmax = tf.math.minimum(bb_x_max, tf.transpose(gt_x_max, [0, 2, 1]))
    i_ymin = tf.math.maximum(bb_y_min, tf.transpose(gt_y_min, [0, 2, 1]))
    i_ymax = tf.math.minimum(bb_y_max, tf.transpose(gt_y_max, [0, 2, 1]))
    i_area = tf.math.maximum((i_xmax - i_xmin), 0) * tf.math.maximum(
        (i_ymax - i_ymin), 0)

    # Calculates the union area.
    bb_area = (bb_y_max - bb_y_min) * (bb_x_max - bb_x_min)
    gt_area = (gt_y_max - gt_y_min) * (gt_x_max - gt_x_min)
    # Adds a small epsilon to avoid divide-by-zero.
    u_area = bb_area + tf.transpose(gt_area, [0, 2, 1]) - i_area + 1e-8

    # Calculates IoU.
    iou = i_area / u_area

    # Fills -1 for IoU entries between the padded ground truth boxes.
    gt_invalid_mask = tf.less(
        tf.reduce_max(gt_boxes, axis=-1, keepdims=True), 0.0)
    padding_mask = tf.logical_or(
        tf.zeros_like(bb_x_min, dtype=tf.bool),
        tf.transpose(gt_invalid_mask, [0, 2, 1]))
    iou = tf.where(padding_mask, -tf.ones_like(iou), iou)

    return iou

def ragged_box_matching(ragged_boxes, ragged_gt_boxes):
    ''' same as box_matching, but take ragged tensor as input and output ragged tensors'''

    # boxes = ragged_boxes.to_tensor(-1, shape=[None, max_boxes, 4])
    # gt_boxes = ragged_gt_boxes.to_tensor(-1, shape=[None, max_gt_boxes, 4])
    boxes = ragged_boxes.to_tensor(-1)
    gt_boxes = ragged_gt_boxes.to_tensor(-1)
    matched_boxes, matched_indices, matched_ious, ious = box_matching(boxes, gt_boxes)

    unpadding_indices = tf.where(boxes[:,:,0] != -1)
    row_starts = ragged_boxes.row_starts()
    matched_boxes = tf.RaggedTensor.from_row_starts(tf.gather_nd(matched_boxes, unpadding_indices), row_starts)
    matched_indices = tf.RaggedTensor.from_row_starts(tf.gather_nd(matched_indices, unpadding_indices), row_starts)
    matched_ious= tf.RaggedTensor.from_row_starts(tf.gather_nd(matched_ious, unpadding_indices), row_starts)

    return matched_boxes, matched_indices, matched_ious, ious

def mask_matching(proposals, ragged_gt_mask_indices):
    def match_fn(inputs):
        p,mi = inputs
        ious = proposal_iou(p, mi)
        matched_ious = tf.reduce_max(ious, axis=-1)
        matched_indices = tf.argmax(ious, axis=-1)
        return matched_indices, matched_ious

    matched_indices, matched_ious = tf.map_fn(
        match_fn,
        [proposals, ragged_gt_mask_indices],
        fn_output_signature=(tf.RaggedTensorSpec((None,), tf.int64, 0), tf.RaggedTensorSpec((None,), tf.float32, 0))
    )
