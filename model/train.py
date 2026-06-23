import tensorflow as tf
import os
from model.retinanet import build_retinanet, NUM_CLASSES
from model.anchors import generate_all_anchors
from model.retinanet_trainer import RetinaNetTrainer

MAX_BOXES  = 14
BATCH_SIZE = 8
EPOCHS     = 30
IMAGE_SIZE = 640

TRAIN_TFRECORD = 'train.tfrecord'
VALID_TFRECORD = 'valid.tfrecord'
WEIGHTS_PATH   = 'checkpoints/retinanet_best.weights.h5'


def parse_tfrecord(example_proto):
    feature_description = {
        'image':  tf.io.FixedLenFeature([], tf.string),
        'bboxes': tf.io.FixedLenFeature([MAX_BOXES * 4], tf.float32),
        'labels': tf.io.FixedLenFeature([MAX_BOXES], tf.int64),
    }
    ex = tf.io.parse_single_example(example_proto, feature_description)

    image = tf.image.decode_jpeg(ex['image'], channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    bboxes = tf.reshape(ex['bboxes'], (MAX_BOXES, 4))
    labels = tf.cast(tf.reshape(ex['labels'], (MAX_BOXES, 1)), tf.float32)

    targets = tf.concat([bboxes, labels], axis=-1)

    return image, targets


def load_dataset(tfrecord_path, batch_size, shuffle=True):
    ds = tf.data.TFRecordDataset(tfrecord_path)
    ds = ds.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_model():
    base_model = build_retinanet(num_classes=NUM_CLASSES)
    anchors    = generate_all_anchors()

    model = RetinaNetTrainer(
        base_model=base_model,
        anchors=anchors,
        num_classes=NUM_CLASSES,
    )
    model.build((None, IMAGE_SIZE, IMAGE_SIZE, 3))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4)
    )
    return model


def build_callbacks():
    os.makedirs('checkpoints', exist_ok=True)

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=WEIGHTS_PATH,
            monitor='val_cls_loss',
            save_best_only=True,
            save_weights_only=True,
            mode='min',
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_cls_loss',
            mode='min',
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_cls_loss',
            mode='min',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            'checkpoints/training_log.csv',
            separator=',',
            append=False,
        ),
    ]


if __name__ == '__main__':
    print("Загружаем датасеты...")
    train_ds = load_dataset(TRAIN_TFRECORD, batch_size=BATCH_SIZE, shuffle=True)
    valid_ds = load_dataset(VALID_TFRECORD, batch_size=BATCH_SIZE, shuffle=False)

    print("Строим модель...")
    model = build_model()

    print("Начинаем обучение...")
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        callbacks=build_callbacks(),
    )

    print(f"\nОбучение завершено. Веса сохранены: {WEIGHTS_PATH}")