import json
import matplotlib.pyplot as plt
from collections import Counter
import statistics
import os


def load_coco(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_split(coco, split_name):
    categories = {c["id"]: c["name"] for c in coco["categories"] if c["supercategory"] != "none"}
    
    annotations = [a for a in coco["annotations"] if a["category_id"] in categories]
    images = coco["images"]
    annotated_ids = {a["image_id"] for a in annotations}
    
    cat_counter = Counter(categories[a["category_id"]] for a in annotations)
    anns_per_image = Counter(a["image_id"] for a in annotations)
    max_boxes = max(anns_per_image.values()) if anns_per_image else 0
    bbox_areas = [a["bbox"][2] * a["bbox"][3] for a in annotations]
    
    print(f"Сплит: {split_name}")
    print(f"  Изображений всего:           {len(images)}")
    print(f"  Изображений без аннотаций:   {len(images) - len(annotated_ids)}")
    print(f"  Аннотаций всего:             {len(annotations)}")
    print(f"  Макс. боксов на изображение: {max_boxes}")
    print(f"  Классы:")
    for name, cnt in cat_counter.most_common():
        print(f"    {name}: {cnt}")
    
    if bbox_areas:
        median_area = statistics.median(bbox_areas)
        print(f"  Медиана площади bbox:        {median_area:.0f} px^2 "
              f"({median_area/(640*640)*100:.1f}% от изображения)")
    
    return max_boxes, bbox_areas, cat_counter


if __name__ == "__main__":
    all_max_boxes = []
    all_areas = []

    for split in ['train', 'valid', 'test']:
        path = f'dataset/{split}/_annotations.coco.json'
        if os.path.exists(path):
            coco = load_coco(path)
            max_b, areas, cats = analyze_split(coco, split)
            all_max_boxes.append(max_b)
            all_areas.extend(areas)

    print(f"\n>>> Глобальный max_boxes для TFRecord: {max(all_max_boxes)}")

    # Гистограмма площадей
    plt.figure(figsize=(9, 4))
    plt.hist(all_areas, bins=50, log=True, color='steelblue', edgecolor='white')
    plt.title("Распределение площадей bounding boxes (все сплиты)")
    plt.xlabel("Площадь (px^2)")
    plt.ylabel("Количество (log)")
    plt.tight_layout()
    plt.savefig("bbox_area_distribution.png", dpi=150)
    plt.show()
