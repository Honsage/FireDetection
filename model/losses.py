import tensorflow as tf


class FocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss для классификации.
    Подавляет лёгкие примеры (фон) и фокусирует обучение
    на сложных (объекты).

    alpha: балансировка классов (0.25 – стандарт RetinaNet)
    gamma: сила подавления лёгких примеров (2.0 – стандарт)
    """
    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )
        probs = tf.sigmoid(y_pred)
        # Вероятность истинного класса для модулирующего фактора
        p_t = y_true * probs + (1 - y_true) * (1 - probs)
        # Альфа-взвешивание
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        # Модулирующий фактор подавляет лёгкие примеры
        modulating = tf.pow(1.0 - p_t, self.gamma)

        loss = alpha_factor * modulating * ce_loss

        num_pos = tf.maximum(tf.reduce_sum(y_true), 1.0)
        return tf.reduce_sum(loss) / num_pos


class SmoothL1Loss(tf.keras.losses.Loss):
    """
    Smooth L1 Loss для регрессии боксов.
    Считается только по позитивным якорям (pos_mask).
    Фоновые якоря не участвуют, иначе градиенты сводят боксы к нулю.

    delta: порог перехода между L1 и L2 (1.0 по стандарту)
    """
    def __init__(self, delta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta

    def call(self, y_true, y_pred, pos_mask=None):
        diff = tf.abs(y_true - y_pred)
        loss = tf.where(
            diff < self.delta,
            0.5 * tf.square(diff),
            diff - 0.5 * self.delta
        )

        if pos_mask is not None:
            mask    = tf.cast(pos_mask, tf.float32)
            mask    = tf.expand_dims(mask, -1)
            loss    = loss * mask
            num_pos = tf.maximum(tf.reduce_sum(mask), 1.0)
            return tf.reduce_sum(loss) / num_pos

        return tf.reduce_mean(loss)
