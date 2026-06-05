import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Конфигурация модели
INPUT_SIZE = 640
NUM_CLASSES = 4
BATCH_SIZE = 8


def build_retinanet(
    num_classes, input_shape=(INPUT_SIZE, INPUT_SIZE, 3), feature_size=256
):
    # Backbone
    backbone_base = tf.keras.applications.ResNet50(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    c3 = backbone_base.get_layer("conv3_block4_out").output  # 80x80
    c4 = backbone_base.get_layer("conv4_block6_out").output  # 40x40
    c5 = backbone_base.get_layer("conv5_block3_out").output  # 20x20
    backbone = tf.keras.Model(inputs=backbone_base.input, outputs=[c3, c4, c5])

    # FPN
    def build_fpn(c3, c4, c5):
        p5_1 = layers.Conv2D(feature_size, 1)(c5)
        p4_1 = layers.Conv2D(feature_size, 1)(c4)
        p3_1 = layers.Conv2D(feature_size, 1)(c3)
        p5_up = layers.UpSampling2D()(p5_1)
        p4_merge = layers.Add()([p5_up, p4_1])
        p4_up = layers.UpSampling2D()(p4_merge)
        p3_merge = layers.Add()([p4_up, p3_1])
        p3 = layers.Conv2D(feature_size, 3, padding="same")(p3_merge)
        p4 = layers.Conv2D(feature_size, 3, padding="same")(p4_merge)
        p5 = layers.Conv2D(feature_size, 3, padding="same")(p5_1)
        p6 = layers.Conv2D(feature_size, 3, strides=2, padding="same")(c5)
        p6_relu = layers.Activation("relu")(p6)
        p7 = layers.Conv2D(feature_size, 3, strides=2, padding="same")(p6_relu)
        return [p3, p4, p5, p6, p7]

    # Классификационная подсеть
    def classification_subnet(num_classes, num_anchors=9):
        inputs = layers.Input(shape=(None, None, feature_size))
        x = inputs
        for _ in range(4):
            x = layers.Conv2D(feature_size, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(num_classes * num_anchors, 3, padding="same")(x)
        x = layers.Reshape((-1, num_classes))(x)
        return tf.keras.Model(inputs=inputs, outputs=x)

    # Регрессионная подсеть
    def regression_subnet(num_anchors=9):
        inputs = layers.Input(shape=(None, None, feature_size))
        x = inputs
        for _ in range(4):
            x = layers.Conv2D(feature_size, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(4 * num_anchors, 3, padding="same")(x)
        x = layers.Reshape((-1, 4))(x)
        return tf.keras.Model(inputs=inputs, outputs=x)

    inputs = backbone.input
    c3, c4, c5 = backbone(inputs)
    features = build_fpn(c3, c4, c5)

    cls_head = classification_subnet(num_classes)
    reg_head = regression_subnet()

    cls_outputs = [cls_head(f) for f in features]
    reg_outputs = [reg_head(f) for f in features]

    cls_output = layers.Concatenate(axis=1)(cls_outputs)
    reg_output = layers.Concatenate(axis=1)(reg_outputs)

    return tf.keras.Model(
        inputs=inputs, outputs=[cls_output, reg_output], name="RetinaNet"
    )


# Создание якорных рамок
def generate_anchors(base_size, scales, ratios):
    anchors = []
    for scale in scales:
        for ratio in ratios:
            w = base_size * scale * np.sqrt(1.0 / ratio)
            h = base_size * scale * np.sqrt(ratio)
            x1 = -w / 2
            y1 = -h / 2
            x2 = w / 2
            y2 = h / 2
            anchors.append([x1, y1, x2, y2])
    return np.array(anchors)


# Размещение рамок
def shift(feature_map_shape, stride, anchors):
    H, W = feature_map_shape
    shift_x = (np.arange(W) + 0.5) * stride
    shift_y = (np.arange(H) + 0.5) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.stack(
        [shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel()], axis=1
    )
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = anchors.reshape((1, A, 4)) + shifts.reshape((K, 1, 4))
    return all_anchors.reshape((-1, 4))


# Генерация якорных рамок для всех уровней FPN
def generate_all_anchors(feature_map_shapes, strides, base_size=32):
    anchors_per_level = []
    scales = [2**0, 2 ** (1 / 3), 2 ** (2 / 3)]
    ratios = [1.0, 2.0, 0.5]
    for shape, stride in zip(feature_map_shapes, strides):
        anchors = generate_anchors(base_size, scales, ratios)
        shifted = shift(shape, stride, anchors)
        anchors_per_level.append(shifted)
    return np.concatenate(anchors_per_level, axis=0)


# Метрика Intersection Over Union
def compute_iou(boxes1, boxes2):
    boxes1 = tf.expand_dims(boxes1, 1)
    boxes2 = tf.expand_dims(boxes2, 0)
    y1 = tf.maximum(boxes1[..., 0], boxes2[..., 0])
    x1 = tf.maximum(boxes1[..., 1], boxes2[..., 1])
    y2 = tf.minimum(boxes1[..., 2], boxes2[..., 2])
    x2 = tf.minimum(boxes1[..., 3], boxes2[..., 3])
    inter = tf.maximum(0.0, y2 - y1) * tf.maximum(0.0, x2 - x1)
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union = area1 + area2 - inter
    return inter / tf.maximum(union, 1e-8)


# Вычисление смещения рамок от bbox
def box_transform(anchors, gt_boxes):
    ay = (anchors[:, 0] + anchors[:, 2]) / 2.0
    ax = (anchors[:, 1] + anchors[:, 3]) / 2.0
    ah = anchors[:, 2] - anchors[:, 0]
    aw = anchors[:, 3] - anchors[:, 1]
    by = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0
    bx = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0
    bh = gt_boxes[:, 2] - gt_boxes[:, 0]
    bw = gt_boxes[:, 3] - gt_boxes[:, 1]
    ty = (by - ay) / ah
    tx = (bx - ax) / aw
    th = tf.math.log(bh / ah)
    tw = tf.math.log(bw / aw)
    return tf.stack([ty, tx, th, tw], axis=1)


# Кодирование целевых значений
def encode_targets(
    anchors,
    gt_boxes,
    gt_labels,
    num_classes,
    iou_threshold_pos=0.5,
    iou_threshold_neg=0.4,
):
    num_gt = tf.shape(gt_boxes)[0]

    def encode_with_gt():
        iou_matrix = compute_iou(anchors, gt_boxes)
        matched_iou = tf.reduce_max(iou_matrix, axis=1)
        matched_idx = tf.argmax(iou_matrix, axis=1, output_type=tf.int32)

        cls_targets = tf.one_hot(
            tf.fill([tf.shape(anchors)[0]], num_classes - 1), depth=num_classes
        )
        reg_targets = tf.zeros_like(anchors, dtype=tf.float32)

        pos_mask = matched_iou >= iou_threshold_pos
        pos_indices = tf.where(pos_mask)[:, 0]
        has_pos = tf.shape(pos_indices)[0] > 0

        def process_positive():
            matched_gt = tf.gather(gt_boxes, tf.gather(matched_idx, pos_indices))
            pos_labels = tf.gather(gt_labels, tf.gather(matched_idx, pos_indices))
            pos_one_hot = tf.one_hot(pos_labels - 1, depth=num_classes)

            cls_targets_updated = tf.tensor_scatter_nd_update(
                cls_targets, tf.expand_dims(pos_indices, axis=1), pos_one_hot
            )
            pos_anchors = tf.gather(anchors, pos_indices)
            reg_values = box_transform(pos_anchors, matched_gt)
            reg_targets_updated = tf.tensor_scatter_nd_update(
                reg_targets, tf.expand_dims(pos_indices, 1), reg_values
            )
            return cls_targets_updated, reg_targets_updated

        def no_positive():
            return cls_targets, reg_targets

        return tf.cond(has_pos, process_positive, no_positive)

    def encode_no_gt():
        cls_targets = tf.one_hot(
            tf.fill([tf.shape(anchors)[0]], num_classes - 1), depth=num_classes
        )
        reg_targets = tf.zeros_like(anchors, dtype=tf.float32)
        return cls_targets, reg_targets

    return tf.cond(num_gt > 0, encode_with_gt, encode_no_gt)


# Метрика Focal Loss
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        ce_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        probs = tf.nn.softmax(y_pred)
        probs_true = tf.reduce_sum(probs * y_true, axis=-1)
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        alpha_factor = tf.reduce_sum(alpha_factor, axis=-1)
        modulating_factor = tf.pow(1.0 - probs_true, self.gamma)
        loss = alpha_factor * modulating_factor * ce_loss
        return tf.reduce_mean(loss)


# Метрика Smooth L1 Loss
class SmoothL1Loss(tf.keras.losses.Loss):
    def __init__(self, delta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta

    def call(self, y_true, y_pred):
        diff = tf.abs(y_true - y_pred)
        less = tf.less(diff, self.delta)
        loss = tf.where(less, 0.5 * tf.square(diff), diff - 0.5)
        return tf.reduce_mean(loss)


# Модель-обертка
class RetinaNetModel(tf.keras.Model):
    def __init__(self, base_model, anchors, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.model = base_model
        self.anchors = tf.constant(anchors, dtype=tf.float32)
        self.num_classes = num_classes
        self.focal_loss = FocalLoss()
        self.reg_loss = SmoothL1Loss()
        self.cls_tracker = tf.keras.metrics.Mean(name="cls_loss")
        self.reg_tracker = tf.keras.metrics.Mean(name="reg_loss")

    def build(self, input_shape):
        self.model.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        return self.model(inputs, training=training)

    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer

    # Описание шага обучения
    def train_step(self, data):
        images, targets_raw = data

        def process_single_target(target):
            mask = tf.greater_equal(target[:, 4], 1)
            valid_targets = tf.boolean_mask(target, mask)

            def has_valid_targets():
                gt_boxes = valid_targets[:, :4]
                gt_labels = tf.cast(valid_targets[:, 4], tf.int32)
                return gt_boxes, gt_labels

            def no_valid_targets():
                gt_boxes = tf.zeros((0, 4), dtype=tf.float32)
                gt_labels = tf.zeros((0,), dtype=tf.int32)
                return gt_boxes, gt_labels

            gt_boxes, gt_labels = tf.cond(
                tf.shape(valid_targets)[0] > 0, has_valid_targets, no_valid_targets
            )
            cls_t, reg_t = encode_targets(
                self.anchors, gt_boxes, gt_labels, self.num_classes
            )
            return cls_t, reg_t

        cls_targets, reg_targets = tf.map_fn(
            process_single_target,
            elems=targets_raw,
            fn_output_signature=(
                tf.TensorSpec(
                    shape=(self.anchors.shape[0], self.num_classes), dtype=tf.float32
                ),
                tf.TensorSpec(shape=(self.anchors.shape[0], 4), dtype=tf.float32),
            ),
        )

        with tf.GradientTape() as tape:
            cls_pred, reg_pred = self.model(images, training=True)
            cls_loss = self.focal_loss(cls_targets, cls_pred)
            reg_loss = self.reg_loss(reg_targets, reg_pred)
            total_loss = cls_loss + reg_loss

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.cls_tracker.update_state(cls_loss)
        self.reg_tracker.update_state(reg_loss)

        return {
            "cls_loss": self.cls_tracker.result(),
            "reg_loss": self.reg_tracker.result(),
        }

    # Описание шага валидации после окончания эпохи
    def test_step(self, data):
        images, targets_raw = data

        def process_single_example(inputs):
            image, target = inputs
            mask = tf.greater_equal(target[:, 4], 1)
            valid_targets = tf.boolean_mask(target, mask)

            def has_valid_targets():
                gt_boxes = valid_targets[:, :4]
                gt_labels = tf.cast(valid_targets[:, 4], tf.int32)
                return gt_boxes, gt_labels

            def no_valid_targets():
                gt_boxes = tf.zeros((0, 4), dtype=tf.float32)
                gt_labels = tf.zeros((0,), dtype=tf.int32)
                return gt_boxes, gt_labels

            gt_boxes, gt_labels = tf.cond(
                tf.shape(valid_targets)[0] > 0, has_valid_targets, no_valid_targets
            )
            cls_t, reg_t = encode_targets(
                self.anchors, gt_boxes, gt_labels, self.num_classes
            )
            return cls_t, reg_t

        cls_targets, reg_targets = tf.map_fn(
            process_single_example,
            (images, targets_raw),
            fn_output_signature=(
                tf.TensorSpec(
                    shape=(self.anchors.shape[0], self.num_classes), dtype=tf.float32
                ),
                tf.TensorSpec(shape=(self.anchors.shape[0], 4), dtype=tf.float32),
            ),
        )

        cls_pred, reg_pred = self.model(images, training=False)
        cls_loss = self.focal_loss(cls_targets, cls_pred)
        reg_loss = self.reg_loss(reg_targets, reg_pred)
        total_loss = cls_loss + reg_loss

        self.cls_tracker.update_state(cls_loss)
        self.reg_tracker.update_state(reg_loss)

        return {
            "loss": total_loss,
            "cls_loss": self.cls_tracker.result(),
            "reg_loss": self.reg_tracker.result(),
        }


# Создание модели
num_classes = 4
base_model = build_retinanet(num_classes)

# Генерация якорных рамок
feature_shapes = [(80, 80), (40, 40), (20, 20), (10, 10), (5, 5)]
strides = [8, 16, 32, 64, 128]
anchors = generate_all_anchors(feature_shapes, strides)

# Оборачивание в кастомную модель
model = RetinaNetModel(base_model, anchors=anchors, num_classes=num_classes)

# Построение модели
model.build((None, 640, 640, 3))


# Десериализация TFRecord
def parse_tfrecord_fn(example_proto, max_boxes=14):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "bboxes": tf.io.FixedLenFeature([max_boxes * 4], tf.float32),
        "labels": tf.io.FixedLenFeature([max_boxes], tf.int64),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)

    image = tf.image.decode_jpeg(example["image"], channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (640, 640))

    bboxes = tf.reshape(example["bboxes"], (max_boxes, 4)) * tf.constant(
        [640.0, 640.0, 640.0, 640.0]
    )
    labels = tf.reshape(example["labels"], (max_boxes, 1))
    targets = tf.concat([bboxes, tf.cast(labels, tf.float32)], axis=-1)

    return image, targets


# Загрузка датасета из TFRecord
def load_tfrecord_dataset(tfrecord_path, batch_size=8, shuffle=True):
    ds = tf.data.TFRecordDataset(tfrecord_path)
    ds = ds.map(lambda x: parse_tfrecord_fn(x), num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = load_tfrecord_dataset("train.tfrecord", batch_size=8)
val_ds = load_tfrecord_dataset("valid.tfrecord", batch_size=8)

# Функции обратного вызова
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath="retinanet_best_weights.weights.h5",
        monitor="val_cls_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_cls_loss",
        mode="min",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_cls_loss", mode="min", factor=0.5, patience=3, verbose=1
    ),
]

# Компиляция модели
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

# Обучение модели
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=12,
    steps_per_epoch=256,
    validation_steps=220,
    callbacks=callbacks,
)
