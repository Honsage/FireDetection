import numpy as np

# Каждый уровень FPN отвечает за свой масштаб объектов:
# P3(stride=8)   - base_size=32
# P4(stride=16)  - base_size=64
# P5(stride=32)  - base_size=128
# P6(stride=64)  - base_size=256
# P7(stride=128) - base_size=512
FPN_BASE_SIZES = [32, 64, 128, 256, 512]
FPN_STRIDES    = [8, 16, 32, 64, 128]
ANCHOR_SCALES  = [2**0, 2**(1/3), 2**(2/3)]
ANCHOR_RATIOS  = [1.0, 2.0, 0.5]


def generate_anchors(base_size, scales, ratios):
    """
    Генерирует 9 якорей для одной ячейки сетки.
    Возвращает [y1, x1, y2, x2] в абсолютных координатах
    относительно центра ячейки (т.е. значения отрицательные/положительные)
    """
    anchors = []
    for scale in scales:
        for ratio in ratios:
            w = base_size * scale * np.sqrt(1.0 / ratio)
            h = base_size * scale * np.sqrt(ratio)
            anchors.append([
                -h / 2,  # y1
                -w / 2,  # x1
                 h / 2,  # y2
                 w / 2,  # x2
            ])
    return np.array(anchors, dtype=np.float32)


def shift_anchors(feature_map_shape, stride, anchors):
    """
    Размещает якоря по всей карте признаков.
    feature_map_shape: (H, W)
    stride: шаг сетки в пикселях исходного изображения
    Возвращает (H*W*9, 4) в абсолютных пикселях [y1,x1,y2,x2]
    """
    H, W = feature_map_shape
    # Центры ячеек в координатах исходного изображения
    shift_y = (np.arange(H) + 0.5) * stride
    shift_x = (np.arange(W) + 0.5) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    # (H*W, 4) — смещения [dy, dx, dy, dx]
    shifts = np.stack([
        shift_y.ravel(),
        shift_x.ravel(),
        shift_y.ravel(),
        shift_x.ravel(),
    ], axis=1).astype(np.float32)

    A = anchors.shape[0]
    K = shifts.shape[0]

    all_anchors = shifts.reshape(K, 1, 4) + anchors.reshape(1, A, 4)
    return all_anchors.reshape(-1, 4)


def generate_all_anchors(feature_map_shapes=None):
    """
    Генерирует якоря для всех 5 уровней FPN.
    feature_map_shapes: список (H, W) для P3..P7
                        по умолчанию для input 640x640
    Возвращает np.array (N, 4) в абсолютных пикселях [y1,x1,y2,x2]
    где N = 80*80*9 + 40*40*9 + 20*20*9 + 10*10*9 + 5*5*9 = 76725
    """
    if feature_map_shapes is None:
        feature_map_shapes = [(80, 80), (40, 40), (20, 20), (10, 10), (5, 5)]

    all_anchors = []
    for shape, stride, base_size in zip(
        feature_map_shapes, FPN_STRIDES, FPN_BASE_SIZES
    ):
        anchors = generate_anchors(base_size, ANCHOR_SCALES, ANCHOR_RATIOS)
        shifted = shift_anchors(shape, stride, anchors)
        all_anchors.append(shifted)

    return np.concatenate(all_anchors, axis=0)


if __name__ == "__main__":
    anchors = generate_all_anchors()
    print(f"Всего якорей: {anchors.shape[0]}")
    print(f"Формат: {anchors.shape}")
    print(f"Диапазон координат: [{anchors.min():.1f}, {anchors.max():.1f}]")
    print(f"\nПервые 3 якоря (P3, первая ячейка):\n{anchors[:3]}")

    sizes_per_level = []
    level_sizes = [80*80*9, 40*40*9, 20*20*9, 10*10*9, 5*5*9]
    start = 0
    for i, count in enumerate(level_sizes):
        level_anchors = anchors[start:start+count]
        h = level_anchors[:, 2] - level_anchors[:, 0]
        w = level_anchors[:, 3] - level_anchors[:, 1]
        area = np.sqrt(h * w)
        print(f"P{i+3} (stride={FPN_STRIDES[i]}): "
              f"median size = {np.median(area):.1f}px")
        start += count
