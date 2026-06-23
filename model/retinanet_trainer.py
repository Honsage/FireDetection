import tensorflow as tf
from model.losses import FocalLoss, SmoothL1Loss
from model.encoder import encode_targets


class RetinaNetTrainer(tf.keras.Model):
    """
    Обёртка над базовой моделью RetinaNet.
    Реализует кастомные train_step и test_step с:
      - передачей pos_mask в SmoothL1Loss
      - фильтрацией ignore_mask из FocalLoss
      - нормировкой лоссов на num_pos
    """

    def __init__(self, base_model, anchors, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.model       = base_model
        self.anchors     = tf.constant(anchors, dtype=tf.float32)
        self.num_classes = num_classes

        self.focal_loss  = FocalLoss(alpha=0.25, gamma=2.0)
        self.reg_loss    = SmoothL1Loss(delta=1.0)

        # Метрики отслеживаем отдельно для удобства мониторинга
        self.cls_metric   = tf.keras.metrics.Mean(name='cls_loss')
        self.reg_metric   = tf.keras.metrics.Mean(name='reg_loss')
        self.total_metric = tf.keras.metrics.Mean(name='total_loss')

    def build(self, input_shape):
        self.model.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        return self.model(inputs, training=training)

    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer

    def _encode_single(self, target):
        """
        target: (MAX_BOXES, 5) - [y1, x1, y2, x2, label]
        """
        # Фильтруем паддинг (label=0)
        mask         = target[:, 4] >= 1
        valid        = tf.boolean_mask(target, mask)

        def has_gt():
            gt_boxes  = valid[:, :4]
            gt_labels = tf.cast(valid[:, 4], tf.int32)
            return gt_boxes, gt_labels

        def no_gt():
            return (tf.zeros((0, 4), tf.float32),
                    tf.zeros((0,),   tf.int32))

        gt_boxes, gt_labels = tf.cond(
            tf.shape(valid)[0] > 0, has_gt, no_gt
        )


        IMAGE_SIZE = 640.0
        gt_boxes_abs = gt_boxes * IMAGE_SIZE

        cls_t, reg_t, pos_mask, ignore_mask = encode_targets(
            self.anchors, gt_boxes_abs, gt_labels, self.num_classes
        )
        return cls_t, reg_t, pos_mask, ignore_mask

    def _encode_batch(self, targets_raw):
        """
        Кодирует весь батч через tf.map_fn.
        targets_raw: (batch, MAX_BOXES, 5)
        """
        num_anchors = self.anchors.shape[0]

        cls_targets, reg_targets, pos_masks, ignore_masks = tf.map_fn(
            self._encode_single,
            elems=targets_raw,
            fn_output_signature=(
                tf.TensorSpec((num_anchors, self.num_classes), tf.float32),
                tf.TensorSpec((num_anchors, 4),                tf.float32),
                tf.TensorSpec((num_anchors,),                  tf.bool),
                tf.TensorSpec((num_anchors,),                  tf.bool),
            )
        )
        return cls_targets, reg_targets, pos_masks, ignore_masks

    def _compute_losses(self, cls_targets, reg_targets,
                        pos_masks, ignore_masks,
                        cls_pred, reg_pred):
        """
        Вычисляет cls_loss и reg_loss с учётом масок.

        cls_loss:
          - позитивные якоря: учитываются с правильным классом
          - негативные (фон): учитываются с нулевым таргетом
          - ignore_mask: полностью исключаются из лосса

        reg_loss:
          - только позитивные якоря (pos_mask)
        """

        valid_mask = ~ignore_masks

        # Обнуляем предсказания и таргеты для игнорируемых якорей
        valid_mask_f = tf.cast(
            tf.expand_dims(valid_mask, -1), tf.float32
        )

        cls_loss = self.focal_loss(
            cls_targets * valid_mask_f,
            cls_pred    * valid_mask_f
        )

        # Регрессионный лосс — только позитивные якоря
        reg_loss = self.reg_loss(
            reg_targets, reg_pred, pos_mask=pos_masks
        )

        return cls_loss, reg_loss

    def train_step(self, data):
        images, targets_raw = data
        # targets_raw: (batch, MAX_BOXES, 5) - [y1,x1,y2,x2, label]

        cls_targets, reg_targets, pos_masks, ignore_masks = \
            self._encode_batch(targets_raw)

        with tf.GradientTape() as tape:
            cls_pred, reg_pred = self.model(images, training=True)
            cls_loss, reg_loss = self._compute_losses(
                cls_targets, reg_targets,
                pos_masks, ignore_masks,
                cls_pred, reg_pred
            )
            total_loss = cls_loss + reg_loss

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables)
        )

        self.cls_metric.update_state(cls_loss)
        self.reg_metric.update_state(reg_loss)
        self.total_metric.update_state(total_loss)

        return {
            'total_loss': self.total_metric.result(),
            'cls_loss':   self.cls_metric.result(),
            'reg_loss':   self.reg_metric.result(),
        }

    def test_step(self, data):
        images, targets_raw = data

        cls_targets, reg_targets, pos_masks, ignore_masks = \
            self._encode_batch(targets_raw)

        cls_pred, reg_pred = self.model(images, training=False)
        cls_loss, reg_loss = self._compute_losses(
            cls_targets, reg_targets,
            pos_masks, ignore_masks,
            cls_pred, reg_pred
        )
        total_loss = cls_loss + reg_loss

        self.cls_metric.update_state(cls_loss)
        self.reg_metric.update_state(reg_loss)
        self.total_metric.update_state(total_loss)

        return {
            'total_loss': self.total_metric.result(),
            'cls_loss':   self.cls_metric.result(),
            'reg_loss':   self.reg_metric.result(),
        }

    @property
    def metrics(self):
        return [self.total_metric, self.cls_metric, self.reg_metric]
