import tensorflow as tf


def compute_iou(anchors, gt_boxes):
    """
    Вычисляет матрицу IoU между якорями и gt_boxes.
    anchors:  (N, 4) [y1, x1, y2, x2]
    gt_boxes: (M, 4) [y1, x1, y2, x2]
    Возвращает: (N, M)
    """
    a = tf.expand_dims(anchors, 1)
    b = tf.expand_dims(gt_boxes, 0)

    # Пересечение
    inter_y1 = tf.maximum(a[..., 0], b[..., 0])
    inter_x1 = tf.maximum(a[..., 1], b[..., 1])
    inter_y2 = tf.minimum(a[..., 2], b[..., 2])
    inter_x2 = tf.minimum(a[..., 3], b[..., 3])

    inter_h = tf.maximum(0.0, inter_y2 - inter_y1)
    inter_w = tf.maximum(0.0, inter_x2 - inter_x1)
    inter_area = inter_h * inter_w

    area_a = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])
    area_b = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])

    union = (tf.expand_dims(area_a, 1) +
             tf.expand_dims(area_b, 0) - inter_area)

    return inter_area / tf.maximum(union, 1e-8)


def box_transform(anchors, gt_boxes):
    """
    Кодирует gt_boxes как смещения относительно якорей.
    Стандартное параметрическое кодирование RetinaNet.
    anchors:  (N, 4) [y1, x1, y2, x2]
    gt_boxes: (N, 4) [y1, x1, y2, x2]
    Возвращает: (N, 4) [ty, tx, th, tw]
    """
    # Центры и размеры якорей
    a_h = anchors[:, 2] - anchors[:, 0]
    a_w = anchors[:, 3] - anchors[:, 1]
    a_cy = anchors[:, 0] + 0.5 * a_h
    a_cx = anchors[:, 1] + 0.5 * a_w

    # Центры и размеры gt
    b_h = gt_boxes[:, 2] - gt_boxes[:, 0]
    b_w = gt_boxes[:, 3] - gt_boxes[:, 1]
    b_cy = gt_boxes[:, 0] + 0.5 * b_h
    b_cx = gt_boxes[:, 1] + 0.5 * b_w

    # Смещения
    ty = (b_cy - a_cy) / tf.maximum(a_h, 1e-8)
    tx = (b_cx - a_cx) / tf.maximum(a_w, 1e-8)
    th = tf.math.log(tf.maximum(b_h / tf.maximum(a_h, 1e-8), 1e-8))
    tw = tf.math.log(tf.maximum(b_w / tf.maximum(a_w, 1e-8), 1e-8))

    return tf.stack([ty, tx, th, tw], axis=1)


def encode_targets(anchors, gt_boxes, gt_labels, num_classes,
                   iou_pos=0.5, iou_neg=0.4):
    """
    Кодирует цели для одного изображения.

    anchors:    (N, 4) tf.float32 [y1,x1,y2,x2] абс. пиксели
    gt_boxes:   (M, 4) tf.float32 [y1,x1,y2,x2] абс. пиксели
    gt_labels:  (M,)   tf.int32   значения 1,2,3
    num_classes: int   = 3

    Возвращает:
        cls_targets: (N, num_classes) float32 – one-hot или нули (фон)
        reg_targets: (N, 4)           float32 – смещения для позитивных
        pos_mask:    (N,)             bool    – True для позитивных якорей
        ignore_mask: (N,)             bool    – True для игнорируемых якорей
    """
    num_anchors = tf.shape(anchors)[0]

    cls_targets = tf.zeros([num_anchors, num_classes], dtype=tf.float32)
    reg_targets = tf.zeros([num_anchors, 4], dtype=tf.float32)
    pos_mask    = tf.zeros([num_anchors], dtype=tf.bool)
    ignore_mask = tf.zeros([num_anchors], dtype=tf.bool)

    def encode_with_gt():
        iou_matrix  = compute_iou(anchors, gt_boxes)
        matched_iou = tf.reduce_max(iou_matrix, axis=1)
        matched_idx = tf.argmax(iou_matrix, axis=1,
                                output_type=tf.int32)

        pos_mask_    = matched_iou >= iou_pos
        ignore_mask_ = (matched_iou >= iou_neg) & ~pos_mask_

        best_anchor_per_gt = tf.argmax(iou_matrix, axis=0,
                                       output_type=tf.int32)
        pos_mask_ = tf.tensor_scatter_nd_update(
            tf.cast(pos_mask_, tf.int32),
            tf.expand_dims(best_anchor_per_gt, 1),
            tf.ones([tf.shape(gt_boxes)[0]], dtype=tf.int32)
        )
        pos_mask_ = tf.cast(pos_mask_, tf.bool)

        # Классификационные цели для позитивных якорей
        pos_indices  = tf.cast(tf.where(pos_mask_), tf.int32)[:, 0]
        pos_gt_idx   = tf.gather(matched_idx, pos_indices)
        pos_labels   = tf.gather(gt_labels, pos_gt_idx)

        pos_one_hot  = tf.one_hot(pos_labels - 1,
                                  depth=num_classes,
                                  dtype=tf.float32)
        cls_targets_ = tf.tensor_scatter_nd_update(
            cls_targets,
            tf.expand_dims(pos_indices, 1),
            pos_one_hot
        )

        # Регрессионные цели – только позитивные
        pos_anchors  = tf.gather(anchors, pos_indices)
        pos_gt_boxes = tf.gather(gt_boxes, pos_gt_idx)
        reg_values   = box_transform(pos_anchors, pos_gt_boxes)
        reg_targets_ = tf.tensor_scatter_nd_update(
            reg_targets,
            tf.expand_dims(pos_indices, 1),
            reg_values
        )

        return cls_targets_, reg_targets_, pos_mask_, ignore_mask_

    def encode_no_gt():
        return cls_targets, reg_targets, pos_mask, ignore_mask

    return tf.cond(
        tf.shape(gt_boxes)[0] > 0,
        encode_with_gt,
        encode_no_gt
    )
