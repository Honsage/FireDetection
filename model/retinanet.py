import tensorflow as tf
from tensorflow.keras import layers


INPUT_SIZE   = 640
NUM_CLASSES  = 3
FEATURE_SIZE = 256


def build_fpn(c3, c4, c5, feature_size=FEATURE_SIZE):
    """
    Feature Pyramid Network.
    Входы:  c3 (80x80), c4 (40x40), c5 (20x20) – из ResNet50
    Выходы: [p3, p4, p5, p6, p7]
    """
    # Lateral connections – приводим все к feature_size каналам
    p5_1 = layers.Conv2D(feature_size, 1, name='fpn_c5')(c5)
    p4_1 = layers.Conv2D(feature_size, 1, name='fpn_c4')(c4)
    p3_1 = layers.Conv2D(feature_size, 1, name='fpn_c3')(c3)

    # Top-down путь
    p5_up     = layers.UpSampling2D(2, name='fpn_up5')(p5_1)
    p4_merge  = layers.Add(name='fpn_add4')([p5_up, p4_1])
    p4_up     = layers.UpSampling2D(2, name='fpn_up4')(p4_merge)
    p3_merge  = layers.Add(name='fpn_add3')([p4_up, p3_1])

    # Output convolutions – сглаживают артефакты upsampling
    p3 = layers.Conv2D(feature_size, 3, padding='same', name='fpn_p3')(p3_merge)
    p4 = layers.Conv2D(feature_size, 3, padding='same', name='fpn_p4')(p4_merge)
    p5 = layers.Conv2D(feature_size, 3, padding='same', name='fpn_p5')(p5_1)

    # P6, P7 – subsampling от c5 для крупных объектов
    p6       = layers.Conv2D(feature_size, 3, strides=2,
                             padding='same', name='fpn_p6')(c5)
    p6_relu  = layers.ReLU(name='fpn_p6_relu')(p6)
    p7       = layers.Conv2D(feature_size, 3, strides=2,
                             padding='same', name='fpn_p7')(p6_relu)

    return [p3, p4, p5, p6, p7]


def build_classification_head(num_classes, num_anchors=9,
                               feature_size=FEATURE_SIZE):
    """
    Классификационная подсеть – общая для всех уровней FPN.
    4 свёрточных слоя + финальный возвращают (num_anchors * num_classes) на ячейку
    """
    inputs = layers.Input(shape=(None, None, feature_size))
    x = inputs
    for i in range(4):
        x = layers.Conv2D(feature_size, 3, padding='same',
                          activation='relu',
                          name=f'cls_conv{i}')(x)
    x = layers.Conv2D(num_anchors * num_classes, 3,
                      padding='same', name='cls_output')(x)

    x = layers.Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], -1, num_classes]),
        name='cls_reshape'
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=x,
                          name='classification_head')


def build_regression_head(num_anchors=9, feature_size=FEATURE_SIZE):
    """
    Регрессионная подсеть – общая для всех уровней FPN.
    4 свёрточных слоя + финальный возвращают (num_anchors * 4) на ячейку
    """
    inputs = layers.Input(shape=(None, None, feature_size))
    x = inputs
    for i in range(4):
        x = layers.Conv2D(feature_size, 3, padding='same',
                          activation='relu',
                          name=f'reg_conv{i}')(x)
    x = layers.Conv2D(num_anchors * 4, 3,
                      padding='same', name='reg_output')(x)

    x = layers.Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]),
        name='reg_reshape'
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=x,
                          name='regression_head')


def build_retinanet(num_classes=NUM_CLASSES,
                    input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
                    feature_size=FEATURE_SIZE):
    """
    Собирает полную модель RetinaNet.
    Возвращает tf.keras.Model с выходами:
        cls_output: (batch, 76725, num_classes) – логиты классификации
        reg_output: (batch, 76725, 4)           – смещения боксов
    """
    # Backbone – ResNet50 без верхних слоёв
    backbone_base = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    # Извлекаем выходы трёх уровней
    c3 = backbone_base.get_layer('conv3_block4_out').output  # 80x80
    c4 = backbone_base.get_layer('conv4_block6_out').output  # 40x40
    c5 = backbone_base.get_layer('conv5_block3_out').output  # 20x20

    backbone = tf.keras.Model(
        inputs=backbone_base.input,
        outputs=[c3, c4, c5],
        name='backbone'
    )

    # FPN
    inputs       = backbone.input
    c3, c4, c5   = backbone(inputs)
    features     = build_fpn(c3, c4, c5, feature_size)

    # Головы – создаются один раз, веса разделяются между уровнями FPN
    cls_head = build_classification_head(num_classes)
    reg_head = build_regression_head()

    # Применяем головы к каждому уровню FPN
    cls_outputs = [cls_head(f) for f in features]
    reg_outputs = [reg_head(f) for f in features]

    # Concatenate по оси якорей
    cls_output = layers.Concatenate(axis=1, name='cls_concat')(cls_outputs)
    reg_output = layers.Concatenate(axis=1, name='reg_concat')(reg_outputs)

    return tf.keras.Model(
        inputs=inputs,
        outputs=[cls_output, reg_output],
        name='RetinaNet'
    )


if __name__ == "__main__":
    model = build_retinanet()
    model.summary(line_length=80)

    import numpy as np
    dummy = np.zeros((2, 640, 640, 3), dtype=np.float32)
    cls_out, reg_out = model(dummy, training=False)
    print(f"\ncls_output shape: {cls_out.shape}")
    print(f"reg_output shape: {reg_out.shape}")
