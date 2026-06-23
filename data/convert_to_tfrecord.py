import json
import os
import tensorflow as tf
from collections import defaultdict


REAL_CATEGORY_IDS = {1, 2, 3}
MAX_BOXES = 14
IMAGE_SIZE = 640


def coco_to_tfrecord(coco_json_path, images_dir, output_path):
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}

    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["category_id"] in REAL_CATEGORY_IDS:
            anns_by_image[ann["image_id"]].append(ann)

    writer = tf.io.TFRecordWriter(output_path)
    skipped = 0
    written = 0

    for img_id, img_info in images.items():
        img_path = os.path.join(images_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            skipped += 1
            continue

        with open(img_path, 'rb') as f:
            image_bytes = f.read()

        W = img_info["width"]
        H = img_info["height"]
        anns = anns_by_image[img_id]  # пусто = negative sample

        bboxes_flat = []
        labels_flat = []

        for ann in anns[:MAX_BOXES]:
            x, y, w, h = ann["bbox"]
            # COCO [x, y, w, h] px в [y1, x1, y2, x2] normalized
            y1 = max(0.0, y / H)
            x1 = max(0.0, x / W)
            y2 = min(1.0, (y + h) / H)
            x2 = min(1.0, (x + w) / W)
            bboxes_flat.extend([y1, x1, y2, x2])
            labels_flat.append(ann["category_id"])

        # Паддинг нулями до MAX_BOXES
        pad = MAX_BOXES - len(anns[:MAX_BOXES])
        bboxes_flat.extend([0.0] * (pad * 4))
        labels_flat.extend([0] * pad)

        feature = {
            "image":  tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[image_bytes])),
            "bboxes": tf.train.Feature(
                float_list=tf.train.FloatList(value=bboxes_flat)),
            "labels": tf.train.Feature(
                int64_list=tf.train.Int64List(value=labels_flat)),
        }
        example = tf.train.Example(
            features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
        written += 1

    writer.close()
    print(f"  Записано:  {written}")
    print(f"  Пропущено: {skipped} (файл не найден)")
    print(f"  Сохранено: {output_path}")


if __name__ == "__main__":
    for split in ['train', 'valid', 'test']:
        print(f"\nКонвертируем {split}...")
        coco_to_tfrecord(
            coco_json_path=f'dataset/{split}/_annotations.coco.json',
            images_dir=f'dataset/{split}',
            output_path=f'{split}.tfrecord',
        )