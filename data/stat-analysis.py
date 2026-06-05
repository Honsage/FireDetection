import json
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import statistics


def load_coco_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_category_info(coco_data):
    return {cat["id"]: cat["name"] for cat in coco_data["categories"]}


def analyze_annotations(coco_data):
    annotations = coco_data["annotations"]
    images = coco_data["images"]
    category_names = get_category_info(coco_data)

    annotated_image_ids = {ann["image_id"] for ann in annotations}
    images_without_annotations = [
        img for img in images if img["id"] not in annotated_image_ids
    ]

    category_counter = Counter()
    for ann in annotations:
        category_counter[ann["category_id"]] += 1

    category_counter_named = {category_names[k]: v for k, v in category_counter.items()}

    return {
        "num_images": len(images),
        "num_annotations": len(annotations),
        "num_images_without_annotations": len(images_without_annotations),
        "category_counts": category_counter_named,
        "images_without_annotations": images_without_annotations,
    }


def print_report(stats, sample):
    print(f"\nОбщая статистика по {sample}:")
    print(f"Изображений: {stats['num_images']}")
    print(f"Изображений без аннотаций: {stats['num_images_without_annotations']}")
    print(f"Аннотаций: {stats['num_annotations']}")
    print("Кол-во аннотаций по категориям:")
    for category, count in stats["category_counts"].items():
        print(f" - {category}: {count}")


# Анализ train, test, valid выборок
for sample in ["train", "test", "valid"]:
    coco_data = load_coco_json(f"./dataset/{sample}/_annotations.coco.json")
    stats = analyze_annotations(coco_data)
    print_report(stats, sample)

# Детальный анализ train выборки
with open("dataset/train/_annotations.coco.json", "r", encoding="utf-8") as f:
    coco = json.load(f)

images = {img["id"]: img for img in coco["images"]}
categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
annotations = coco["annotations"]

image_to_anns = defaultdict(list)
class_counter = Counter()
combo_counter = Counter()
bbox_areas = []

for ann in annotations:
    image_to_anns[ann["image_id"]].append(ann)
    class_counter[ann["category_id"]] += 1
    bbox = ann["bbox"]
    area = bbox[2] * bbox[3]
    bbox_areas.append(area)

for img_id, anns in image_to_anns.items():
    class_combo = tuple(sorted(set(ann["category_id"] for ann in anns)))
    combo_counter[class_combo] += 1

# Визуализация сочетаний классов
plt.figure(figsize=(6, 6))
labels = [", ".join(categories[c] for c in k) for k in combo_counter.keys()]
sizes = list(combo_counter.values())
plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=30)
plt.title("Сочетания классов на изображениях")
plt.axis("equal")
plt.show()

# Визуализация площадей bounding boxes
plt.figure(figsize=(8, 4))
plt.hist(bbox_areas, bins=40, log=True, color="skyblue")
plt.title("Площади bounding boxes")
plt.xlabel("Площадь (px²)")
plt.ylabel("Количество (логарифмическая шкала)")
plt.show()

median_area = statistics.median(bbox_areas)
image_area = 640 * 640
print(f"Медианная площадь bbox: {median_area}")
print(f"Медианное отношение площадей: {median_area / image_area}")
