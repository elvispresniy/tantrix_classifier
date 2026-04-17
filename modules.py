from ultralytics import SAM
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

COLOR_CONFIG = {
    'yellow': {
        'hue_ranges': [(15, 40)],       # Hue диапазон в OpenCV (0-180)
        'min_saturation': 30,            # минимальная насыщенность
        'canonical': np.array([0, 255, 0]),  # зелёный BGR для визуальной разделимости
    },
    'red': {
        'hue_ranges': [(0, 15), (165, 180)],  # красный — два диапазона (обёртка)
        'min_saturation': 30,
        'canonical': np.array([0, 0, 255]),    # красный BGR
    },
}

# ============================================================
# PATH_MAP — карта путей для каждого шаблона
# Грани нумеруются с 0 (верх-право) по часовой стрелке
# Шаблоны в исходном положении: вершина строго вверх
# ============================================================

PATH_MAP = {
    "hexagon_1":  {"yellow": [0,0,1,1,0,0], "red": [0,1,0,0,0,1], "blue": [1,0,0,0,1,0]},
    "hexagon_2":  {"yellow": [1,0,0,0,0,1], "red": [0,0,1,1,0,0], "blue": [0,1,0,0,1,0]},
    "hexagon_3":  {"yellow": [1,0,0,0,0,1], "red": [0,1,1,0,0,0], "blue": [0,0,0,1,1,0]},
    "hexagon_4":  {"yellow": [0,0,0,1,0,1], "red": [1,0,1,0,0,0], "blue": [0,1,0,0,1,0]},
    "hexagon_5":  {"yellow": [1,0,0,0,0,1], "red": [0,1,0,0,1,0], "blue": [0,0,1,1,0,0]},
    "hexagond_6": {"yellow": [0,1,0,0,1,0], "red": [0,0,0,1,0,1], "blue": [1,0,1,0,0,0]},
    "hexagond_7": {"yellow": [1,0,0,0,1,0], "red": [0,1,0,0,0,1], "blue": [0,0,1,1,0,0]},
    "hexagon_8":  {"yellow": [0,1,0,0,0,1], "red": [1,0,0,0,1,0], "blue": [0,0,1,1,0,0]},
    "hexagon_9":  {"yellow": [0,0,0,1,0,1], "red": [0,1,0,0,1,0], "blue": [1,0,1,0,0,0]},
    "hexagon_10": {"yellow": [0,0,1,1,0,0], "red": [1,0,0,0,1,0], "blue": [0,1,0,0,0,1]},
}

# Углы от центра к середине каждой грани (math: ось X вправо, ось Y вверх)
# Грань 0: верх-право (60°), 1: право (0°), 2: низ-право (-60°),
# 3: низ-лево (-120°), 4: лево (180°), 5: верх-лево (120°)
EDGE_ANGLES_DEG = [60, 0, -60, -120, 180, 120]
EDGE_LABELS = ['0:TR', '1:R', '2:BR', '3:BL', '4:L', '5:TL']
COLOR_PLT = {'red': 'red', 'yellow': 'orange', 'blue': 'dodgerblue'}

def gray_world_normalize_bgra(patch_bgra):
    bgr = patch_bgra[:, :, :3].astype(np.float32)
    alpha = patch_bgra[:, :, 3]
    mask = alpha > 128

    if mask.sum() == 0:
        return patch_bgra.copy()

    pixels = bgr[mask]
    mean_bgr = pixels.mean(axis=0)
    mean_gray = mean_bgr.mean()

    scale = mean_gray / (mean_bgr + 1e-6)
    bgr_norm = bgr * scale
    bgr_norm = np.clip(bgr_norm, 0, 255).astype(np.uint8)

    out = np.dstack([bgr_norm, alpha])
    return out

def _classify_by_canonical(bgr_pixels, color_config, tolerance=50):
    """
    Классифицирует пиксели по близости к каноничным цветам (canonical).
    Используется после нормализации, когда пиксели уже заменены на canonical.

    Параметры:
    ----------
    bgr_pixels : np.ndarray (N, 3) — BGR пиксели
    color_config : dict — конфигурация с полем 'canonical'
    tolerance : int — максимальное расстояние до canonical для классификации

    Возвращает:
    ----------
    np.ndarray of object (N,) — имена цветов или 'background'
    """
    result = np.full(len(bgr_pixels), 'background', dtype=object)
    min_dist = np.full(len(bgr_pixels), np.inf)

    pixels_f = bgr_pixels.astype(np.float32)

    for color_name, cfg in color_config.items():
        canonical = cfg['canonical'].astype(np.float32)

        # Евклидово расстояние до canonical
        diff = pixels_f - canonical
        dist = np.sqrt((diff ** 2).sum(axis=1))

        # Пиксель принадлежит этому цвету если dist < tolerance и это ближайший
        is_close = dist < tolerance
        is_closer = dist < min_dist

        update = is_close & is_closer
        result[update] = color_name
        min_dist[update] = dist[update]

    return result

def compare_hex_patches(patch1_rgba, patch2_rgba, color_config=COLOR_CONFIG,
                        canonical_tolerance=50,
                        w_same_color=2.0,
                        w_diff_color=3.0,
                        w_color_vs_transparent=0.75):
    """
    Сравнивает два нормализованных RGBA патча по цветовым классам.

    После нормализации пиксели либо цветные (canonical), либо прозрачные.

    Коэффициенты:
    - w_same_color:           бонус за совпадение цветов
    - w_diff_color:           штраф за разные цвета
    - w_color_vs_transparent: штраф за цвет vs прозрачный
    """
    mask1 = patch1_rgba[:, :, 3] > 128
    mask2 = patch2_rgba[:, :, 3] > 128

    union = mask1 | mask2
    union_count = union.sum()

    if union_count < 10:
        return 0.0

    ys, xs = np.where(union)

    colors1 = np.full(len(ys), 'transparent', dtype=object)
    colors2 = np.full(len(ys), 'transparent', dtype=object)

    opaque1 = mask1[ys, xs]
    if opaque1.any():
        bgr1 = patch1_rgba[ys[opaque1], xs[opaque1], :3]
        colors1[opaque1] = _classify_by_canonical(bgr1, color_config, canonical_tolerance)

    opaque2 = mask2[ys, xs]
    if opaque2.any():
        bgr2 = patch2_rgba[ys[opaque2], xs[opaque2], :3]
        colors2[opaque2] = _classify_by_canonical(bgr2, color_config, canonical_tolerance)

    color_names = set(color_config.keys())

    is_color1 = np.array([c in color_names for c in colors1])
    is_color2 = np.array([c in color_names for c in colors2])
    is_transparent1 = (colors1 == 'transparent')
    is_transparent2 = (colors2 == 'transparent')
    same_color = (colors1 == colors2)

    total_score = 0.0
    max_possible = 0.0

    # Цвет vs тот же цвет → бонус
    m = is_color1 & is_color2 & same_color
    count = m.sum()
    total_score += count * w_same_color
    max_possible += count * w_same_color

    # Цвет vs другой цвет → штраф
    m = is_color1 & is_color2 & ~same_color
    count = m.sum()
    total_score -= count * w_diff_color
    max_possible += count * w_diff_color

    # Цвет vs прозрачный → штраф
    m = (is_color1 & is_transparent2) | (is_transparent1 & is_color2)
    count = m.sum()
    total_score -= count * w_color_vs_transparent
    max_possible += count * w_color_vs_transparent

    if max_possible == 0:
        return 0.0

    score = (total_score + max_possible) / (2.0 * max_possible)
    return max(score, 0.0)

def _normalize_hexagond(patch_bgra, tolerance=100, debug=False, name=""):
    """
    Нормализует hexagond шаблон.
    Цвета уже каноничные:
      - Красный: BGR (0, 0, 255) → оставляем
      - Зелёный: BGR (0, 255, 0) → оставляем
      - Чёрный и всё остальное → прозрачный

    tolerance — допуск на интерполяцию при повороте/ресайзе.
    """
    h, w = patch_bgra.shape[:2]
    bgr = patch_bgra[:, :, :3].astype(np.float32)
    alpha = patch_bgra[:, :, 3]
    opaque = alpha > 128

    # Каноничные цвета
    canonical_red = np.array([0, 0, 255], dtype=np.float32)
    canonical_green = np.array([0, 255, 0], dtype=np.float32)

    dist_red = np.sqrt(((bgr - canonical_red) ** 2).sum(axis=2))
    dist_green = np.sqrt(((bgr - canonical_green) ** 2).sum(axis=2))

    is_red = opaque & (dist_red < tolerance)
    is_green = opaque & (dist_green < tolerance)

    # Разрешаем конфликты — ближайший
    conflict = is_red & is_green
    is_red[conflict] = dist_red[conflict] < dist_green[conflict]
    is_green[conflict] = ~is_red[conflict]

    # Создаём нормализованный патч
    normalized = np.zeros_like(patch_bgra)

    # red → canonical red
    normalized[is_red, 0] = 0
    normalized[is_red, 1] = 0
    normalized[is_red, 2] = 255
    normalized[is_red, 3] = 255

    # green → canonical green (= yellow в нашей схеме)
    normalized[is_green, 0] = 0
    normalized[is_green, 1] = 255
    normalized[is_green, 2] = 0
    normalized[is_green, 3] = 255

    if debug:
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))

        axes[0].imshow(cv2.cvtColor(patch_bgra, cv2.COLOR_BGRA2RGBA))
        axes[0].set_title(f"{name}\nОригинал (выровненный)")
        axes[0].axis("off")

        axes[1].imshow(is_red, cmap='Reds')
        axes[1].set_title(f"Red: {is_red.sum()} px")
        axes[1].axis("off")

        axes[2].imshow(is_green, cmap='Greens')
        axes[2].set_title(f"Green (=yellow): {is_green.sum()} px")
        axes[2].axis("off")

        other = opaque & ~is_red & ~is_green
        axes[3].imshow(other, cmap='gray')
        axes[3].set_title(f"Отброшено: {other.sum()} px\n(чёрный контур и пр.)")
        axes[3].axis("off")

        axes[4].imshow(cv2.cvtColor(normalized, cv2.COLOR_BGRA2RGBA))
        axes[4].set_title(f"Нормализован\n{(is_red.sum() + is_green.sum())} px")
        axes[4].axis("off")

        plt.suptitle(f"hexagond нормализация (tolerance={tolerance})", fontsize=12)
        plt.tight_layout()
        plt.show()

    return normalized

def extract_template_patch_normalized(template_rgba, target_size=128, hex_fill_ratio=0.85):
    mask = (template_rgba[:, :, 3] > 128).astype(np.uint8)
    return extract_hex_patch_normalized(template_rgba, mask, target_size, hex_fill_ratio)

def load_class_templates(hexagons_dir, target_size=128, hex_fill_ratio=0.85,
                         debug=False):
    """
    Загружает, нормализует по размеру и цветам шаблоны классов.
    Для файлов hexagond_* — цвета уже нормализованы (красный=BGR(0,0,255), зелёный=BGR(0,255,0)).
    Для файлов hexagon_* — применяется HSV нормализация.
    """
    templates = []
    files = sorted([f for f in os.listdir(hexagons_dir) if f.endswith(".png")])

    for fname in files:
        filepath = os.path.join(hexagons_dir, fname)
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if img is None or img.shape[2] != 4:
            continue

        class_name = os.path.splitext(fname)[0]
        is_hexagond = class_name.startswith("hexagond")

        # Нормализация размера — одинаковая для обоих типов
        patch, center, angle, scale = extract_template_patch_normalized(
            img, target_size, hex_fill_ratio
        )
        if patch is None:
            print(f"  ⚠️ Не удалось нормализовать шаблон {fname}")
            continue

        if is_hexagond:
            # hexagond: цвета уже каноничные, нужно только убрать чёрный фон
            # и оставить только красные (0,0,255) и зелёные (0,255,0) пиксели
            patch_normalized = _normalize_hexagond(patch, debug=debug, name=class_name)
        else:
            # hexagon: обычная HSV нормализация
            patch_normalized, _ = normalize_template_colors(patch, debug=debug)

        templates.append({
            'patch_rgba': patch_normalized,
            'patch_original': patch,
            'class_name': class_name,
            'filepath': filepath,
            'scale': scale,
            'is_hexagond': is_hexagond,
        })

        print(f"  Шаблон {class_name}: scale={scale:.3f} {'(hexagond)' if is_hexagond else ''}")

    return templates

def rotate_patch_60(patch_rgba, times):
    angle = times * 60
    h, w = patch_rgba.shape[:2]
    center = (w / 2, h / 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(patch_rgba, rot_matrix, (w, h),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return rotated

def normalize_template_colors(patch_bgra, color_config=None, debug=False,
                               min_island_area=15, min_island_perimeter=15,
                               min_island_solidity=0.25):
    """
    Нормализует цвета шаблона через HSV Hue.
    Красный и жёлтый различаются по каналу Hue.
    После замены удаляет мелкие и хаотичные островки.
    """
    if color_config is None:
        color_config = COLOR_CONFIG

    patch_bgra = gray_world_normalize_bgra(patch_bgra)

    h, w = patch_bgra.shape[:2]
    bgr = patch_bgra[:, :, :3]
    alpha = patch_bgra[:, :, 3]
    opaque = alpha > 128

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32)    # 0-180
    sat = hsv[:, :, 1].astype(np.float32)    # 0-255

    color_names = list(color_config.keys())

    # === Классификация по Hue ===
    color_mask = np.zeros((h, w), dtype=np.uint8)

    for i, name in enumerate(color_names):
        cfg = color_config[name]
        min_sat = cfg['min_saturation']

        # Объединяем все hue-диапазоны для этого цвета
        hue_match = np.zeros((h, w), dtype=bool)
        for (hue_low, hue_high) in cfg['hue_ranges']:
            hue_match |= (hue >= hue_low) & (hue < hue_high)

        # Пиксель принадлежит цвету если: непрозрачный + hue в диапазоне + насыщенность достаточна
        match = opaque & hue_match & (sat >= min_sat)

        # Не перезаписываем уже назначенные (приоритет первому)
        match = match & (color_mask == 0)
        color_mask[match] = i + 1

    # ============================================================
    # Удаление мелких и хаотичных островков
    # ============================================================
    color_mask_before = color_mask.copy() if debug else None
    total_removed = 0
    removed_details = []

    for i, name in enumerate(color_names):
        class_id = i + 1
        binary = (color_mask == class_id).astype(np.uint8)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        for label_id in range(1, num_labels):
            comp_area = stats[label_id, cv2.CC_STAT_AREA]

            comp_mask = (labels == label_id).astype(np.uint8)
            contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)

            comp_perimeter = 0
            comp_solidity = 0

            if contours:
                cnt = contours[0]
                comp_perimeter = cv2.arcLength(cnt, True)
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                comp_solidity = comp_area / hull_area if hull_area > 0 else 0

            remove = False
            reason = []

            if comp_area < min_island_area:
                remove = True
                reason.append(f"area={comp_area}<{min_island_area}")

            if comp_perimeter < min_island_perimeter:
                remove = True
                reason.append(f"perim={comp_perimeter:.1f}<{min_island_perimeter}")

            if comp_solidity < min_island_solidity:
                remove = True
                reason.append(f"solidity={comp_solidity:.3f}<{min_island_solidity}")

            if remove:
                color_mask[labels == label_id] = 0
                total_removed += comp_area
                removed_details.append({
                    'color': name, 'area': comp_area,
                    'perimeter': comp_perimeter,
                    'solidity': comp_solidity,
                    'reason': ', '.join(reason),
                })

    if total_removed > 0:
        print(f"    Удалено островков: {total_removed} px")
        for rd in removed_details:
            print(f"      {rd['color']:>8s}: area={rd['area']:>4d} "
                  f"perim={rd['perimeter']:>6.1f} "
                  f"solidity={rd['solidity']:.3f} | {rd['reason']}")

    # Нормализованный патч
    normalized = np.zeros_like(patch_bgra)
    for i, name in enumerate(color_names):
        region = color_mask == (i + 1)
        normalized[region, :3] = color_config[name]['canonical']
        normalized[region, 3] = 255

    if debug:
        n_colors = len(color_names)
        has_before = color_mask_before is not None
        n_cols = n_colors + 4 if has_before else n_colors + 3

        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
        col = 0

        # Оригинал GW
        orig_disp = cv2.cvtColor(patch_bgra, cv2.COLOR_BGRA2RGBA)
        axes[col].imshow(orig_disp)
        axes[col].set_title("Оригинал (GW)")
        axes[col].axis("off")
        col += 1

        # Hue канал
        hue_display = hue.copy()
        hue_display[~opaque] = np.nan
        im = axes[col].imshow(hue_display, cmap='hsv', vmin=0, vmax=180)
        axes[col].set_title("Hue (0-180)")
        axes[col].axis("off")
        plt.colorbar(im, ax=axes[col], fraction=0.046)
        col += 1

        # Маска каждого цвета
        for i, name in enumerate(color_names):
            cfg = color_config[name]
            count_after = (color_mask == (i + 1)).sum()
            count_before = (color_mask_before == (i + 1)).sum() if has_before else count_after
            removed = count_before - count_after

            hue_str = ', '.join(f"[{lo}-{hi})" for lo, hi in cfg['hue_ranges'])

            axes[col].imshow(color_mask == (i + 1), cmap='gray')
            axes[col].set_title(
                f"{name} ({count_after} px)\n"
                f"hue: {hue_str}\n"
                f"min_sat: {cfg['min_saturation']}\n"
                f"удалено: {removed} px",
                fontsize=7
            )
            axes[col].axis("off")
            col += 1

        # Удалённые островки
        if has_before:
            diff = (color_mask_before > 0) & (color_mask == 0)
            overlay = cv2.cvtColor(patch_bgra, cv2.COLOR_BGRA2RGBA).copy().astype(np.float32)
            overlay[diff] = overlay[diff] * 0.3 + np.array([255, 0, 0, 255]) * 0.7
            axes[col].imshow(np.clip(overlay, 0, 255).astype(np.uint8))
            axes[col].set_title(f"Удалённые островки\n{total_removed} px", fontsize=8)
            axes[col].axis("off")
            col += 1

        # Нормализованный
        norm_disp = cv2.cvtColor(normalized, cv2.COLOR_BGRA2RGBA)
        axes[col].imshow(norm_disp)
        unclassified = (opaque & (color_mask == 0)).sum()
        total_opaque = opaque.sum()
        axes[col].set_title(f"Нормализован\nне классиф.: {unclassified}/{total_opaque}")
        axes[col].axis("off")

        plt.suptitle(f"HSV нормализация (area≥{min_island_area}, perim≥{min_island_perimeter}, "
                     f"solidity≥{min_island_solidity})", fontsize=11)
        plt.tight_layout()
        plt.show()

    return normalized, color_mask

def extract_hex_patch_normalized(image, mask, target_size=128, hex_fill_ratio=0.85):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return None, None, None, None
    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    pts = approx.reshape(-1, 2).astype(np.float64)
    dists = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    radius = dists.max()
    if radius < 1:
        return None, None, None, None
    angles_from_center = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    diffs = np.abs(angles_from_center - (-np.pi / 2))
    diffs = np.minimum(diffs, 2 * np.pi - diffs)
    top_vertex_angle = angles_from_center[np.argmin(diffs)]
    rotation_angle_rad = top_vertex_angle - (-np.pi / 2)
    rotation_angle_deg = np.degrees(rotation_angle_rad)
    desired_radius = (target_size / 2) * hex_fill_ratio
    scale = desired_radius / radius
    out_cx = target_size / 2
    out_cy = target_size / 2
    cos_a = np.cos(rotation_angle_rad)
    sin_a = np.sin(rotation_angle_rad)
    M_transform = np.array([
        [scale * cos_a, scale * sin_a, -scale * (cx * cos_a + cy * sin_a) + out_cx],
        [-scale * sin_a, scale * cos_a, -scale * (-cx * sin_a + cy * cos_a) + out_cy]
    ], dtype=np.float64)
    if len(image.shape) == 3 and image.shape[2] == 4:
        img_transformed = cv2.warpAffine(image, M_transform, (target_size, target_size),
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    else:
        img_transformed = cv2.warpAffine(image, M_transform, (target_size, target_size),
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    mask_transformed = cv2.warpAffine(mask, M_transform, (target_size, target_size),
                                       flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if len(img_transformed.shape) == 3 and img_transformed.shape[2] == 4:
        img_transformed[:, :, 3] = (mask_transformed * 255).astype(np.uint8)
    elif len(img_transformed.shape) == 3 and img_transformed.shape[2] == 3:
        img_rgba = cv2.cvtColor(img_transformed, cv2.COLOR_BGR2BGRA)
        img_rgba[:, :, 3] = (mask_transformed * 255).astype(np.uint8)
        img_transformed = img_rgba
    return img_transformed, (cx, cy), rotation_angle_deg, scale

def find_hexagon_masks(image_path, model=None, approx_eps=0.02, min_area=500,
                       vertex_range=(5, 8), circularity_range=(0.65, 0.95),
                       relative_area_threshold=0.15, iou_threshold=0.15,
                       contour_area_ratio=0.01, debug=False):
    if model is None:
        model = SAM("sam2.1_b.pt")

    results = model(image_path, conf=0.1)
    masks = results[0].masks.data.cpu().numpy()

    if masks.shape[0] == 0:
        return []

    n_masks = masks.shape[0]
    masks_uint8 = [masks[i].astype(np.uint8) for i in range(n_masks)]
    all_areas = np.array([m.sum() for m in masks_uint8], dtype=np.float64)

    rejected = []
    candidates = []

    for i in range(n_masks):
        mask = masks_uint8[i]
        area = float(all_areas[i])
        info = {
            'mask': mask, 'area': area, 'index': i,
            'vertices': None, 'circularity': None,
            'contour': None, 'score': None, 'max_iou': None,
        }
        if area < min_area:
            info['reject_reason'] = f"Мин. площадь: {area:.0f} < {min_area}"
            rejected.append(info)
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            info['reject_reason'] = "Контур не найден"
            rejected.append(info)
            continue
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            info['reject_reason'] = "Периметр = 0"
            rejected.append(info)
            continue
        approx = cv2.approxPolyDP(contour, approx_eps * perimeter, True)
        n_vertices = len(approx)
        circularity = (4 * np.pi * cv2.contourArea(contour)) / (perimeter ** 2)
        info['vertices'] = n_vertices
        info['circularity'] = circularity
        info['contour'] = approx
        if not (vertex_range[0] <= n_vertices <= vertex_range[1]):
            info['reject_reason'] = f"Вершин: {n_vertices} ∉ [{vertex_range[0]}, {vertex_range[1]}]"
            rejected.append(info)
            continue
        if not (circularity_range[0] <= circularity <= circularity_range[1]):
            info['reject_reason'] = f"Circularity: {circularity:.3f} ∉ [{circularity_range[0]}, {circularity_range[1]}]"
            rejected.append(info)
            continue
        vertex_penalty = abs(n_vertices - 6) / 6.0
        circularity_penalty = abs(circularity - 0.907) / 0.907
        info['score'] = vertex_penalty + circularity_penalty
        candidates.append(info)

    if candidates:
        max_hex_area = max(c['area'] for c in candidates)
        area_cutoff = max_hex_area * relative_area_threshold
        after_area = []
        for c in candidates:
            if c['area'] < area_cutoff:
                c['reject_reason'] = f"Отн. площадь: {c['area']:.0f} < {area_cutoff:.0f}"
                rejected.append(c)
            else:
                after_area.append(c)
    else:
        after_area = []

    after_area.sort(key=lambda x: x['score'])
    final = []
    suppressed_indices = set()
    for i, candidate in enumerate(after_area):
        if i in suppressed_indices:
            continue
        final.append(candidate)
        for j in range(i + 1, len(after_area)):
            if j in suppressed_indices:
                continue
            intersection = np.logical_and(candidate['mask'], after_area[j]['mask']).sum()
            union = np.logical_or(candidate['mask'], after_area[j]['mask']).sum()
            iou = intersection / union if union > 0 else 0.0
            if iou >= iou_threshold:
                after_area[j]['reject_reason'] = f"IoU с маской {candidate['index']}: {iou:.3f}"
                after_area[j]['max_iou'] = iou
                rejected.append(after_area[j])
                suppressed_indices.add(j)

    for i, fi in enumerate(final):
        max_iou = 0.0
        for j, fj in enumerate(final):
            if i == j:
                continue
            intersection = np.logical_and(fi['mask'], fj['mask']).sum()
            union = np.logical_or(fi['mask'], fj['mask']).sum()
            iou = intersection / union if union > 0 else 0.0
            max_iou = max(max_iou, iou)
        fi['max_iou'] = max_iou

    for hm in final:
        mask = hm['mask']
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            continue
        component_areas = stats[1:, cv2.CC_STAT_AREA]
        max_component_area = component_areas.max()
        min_component_area = max_component_area * contour_area_ratio
        cleaned_mask = np.zeros_like(mask)
        for label_id in range(1, num_labels):
            if stats[label_id, cv2.CC_STAT_AREA] >= min_component_area:
                cleaned_mask[labels == label_id] = 1
        hm['mask'] = cleaned_mask
        hm['area'] = float(cleaned_mask.sum())

    if debug and rejected:
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        n_rej = len(rejected)
        cols = min(n_rej, 5)
        rows = (n_rej + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = np.array(axes).flatten() if n_rej > 1 else [axes]
        for j, rm in enumerate(rejected):
            ax = axes[j]
            ax.imshow(img_rgb)
            ax.imshow(rm['mask'], alpha=0.5, cmap="Reds")
            v_str = f"{rm['vertices']}" if rm['vertices'] is not None else "—"
            c_str = f"{rm['circularity']:.3f}" if rm['circularity'] is not None else "—"
            ax.set_title(f"❌ {rm['index']}\nV:{v_str}|C:{c_str}\nA:{rm['area']:.0f}\n{rm['reject_reason']}", fontsize=8, color='red')
            ax.axis("off")
        for k in range(n_rej, len(axes)):
            axes[k].axis("off")
        fig.suptitle("Отфильтрованные маски", fontsize=14, color='red')
        plt.tight_layout()
        plt.show()

    return final


def rotate_path_map(path_map_entry, rotation_steps):
    """
    Поворачивает PATH_MAP на rotation_steps × 60° по часовой.
    new_edges[i] = old_edges[(i + rotation_steps) % 6]
    """
    rotated = {}
    for color, edges in path_map_entry.items():
        n = len(edges)
        shift = rotation_steps % n
        rotated[color] = [edges[(i + shift) % n] for i in range(n)]
    return rotated


def get_rotated_paths(hex_results, path_map=PATH_MAP):
    """Для каждой фишки возвращает PATH_MAP с учётом поворота."""
    rotated_paths = []
    for res in hex_results:
        class_name = res['class_name']
        rotation_deg = res['best_rotation']

        if class_name not in path_map:
            print(f"  ⚠️ {class_name} не найден в PATH_MAP!")
            rotated_paths.append(None)
            continue

        rotation_steps = int(round(rotation_deg / 60)) % 6
        rotated = rotate_path_map(path_map[class_name], rotation_steps)
        rotated_paths.append(rotated)

        print(f"  Tile {len(rotated_paths)}: {class_name}, "
              f"rot={rotation_deg}° ({rotation_steps} steps)")
        for color, edges in rotated.items():
            active = [i for i, v in enumerate(edges) if v == 1]
            print(f"    {color}: {edges} → грани {active}")

    return rotated_paths


def estimate_hex_grid_rotation(hex_results):
    """
    Оценивает угол поворота гексагональной сетки по центрам фишек.
    Для ориентации 'вершина вверху' углы между соседями кратны 60°.
    Возвращает угол отклонения сетки от идеала.
    """
    centers = np.array([(r['center_x'], r['center_y']) for r in hex_results])
    n = len(centers)

    if n < 2:
        return 0.0

    # Расстояния между всеми парами
    all_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            all_dists.append(d)

    min_dist = min(all_dists)
    threshold = min_dist * 1.4

    # Углы к соседям
    angles = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[j, 0] - centers[i, 0]
            dy = centers[j, 1] - centers[i, 1]
            d = np.sqrt(dx**2 + dy**2)
            if d > threshold:
                continue
            angle = np.degrees(np.arctan2(-dy, dx))  # -dy: Y вниз в пикселях
            angles.append(angle)

    if not angles:
        return 0.0

    # Отклонение от ближайшего кратного 60°
    offsets = []
    for a in angles:
        remainder = a % 60
        offset = remainder if remainder <= 30 else remainder - 60
        offsets.append(offset)

    grid_rotation = np.median(offsets)

    print(f"  Углы к соседям: {[f'{a:.1f}°' for a in angles]}")
    print(f"  Отклонения: {[f'{o:.1f}°' for o in offsets]}")
    print(f"  Поворот сетки: {grid_rotation:.1f}°")

    return grid_rotation


def build_adjacency(hex_results):
    """
    Строит граф соседства: какая грань одной фишки прилегает к какой грани другой.
    """
    centers = np.array([(r['center_x'], r['center_y']) for r in hex_results])
    n = len(centers)

    all_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            all_dists.append(np.linalg.norm(centers[i] - centers[j]))

    if not all_dists:
        return {}

    min_dist = min(all_dists)
    threshold = min_dist * 1.4

    edge_angles_rad = [np.radians(a) for a in EDGE_ANGLES_DEG]

    adjacency = {i: {} for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[j, 0] - centers[i, 0]
            dy = centers[j, 1] - centers[i, 1]
            d = np.sqrt(dx**2 + dy**2)

            if d > threshold:
                continue

            angle = np.arctan2(-dy, dx)

            # Ближайшая грань i
            best_edge = 0
            best_diff = float('inf')
            for e, ea in enumerate(edge_angles_rad):
                diff = abs(np.arctan2(np.sin(angle - ea), np.cos(angle - ea)))
                if diff < best_diff:
                    best_diff = diff
                    best_edge = e

            opposite = (best_edge + 3) % 6

            adjacency[i][best_edge] = (j, opposite)
            adjacency[j][opposite] = (i, best_edge)

            print(f"  Tile {i+1} грань {best_edge} ↔ Tile {j+1} грань {opposite} "
                  f"(d={d:.0f}, angle={np.degrees(angle):.1f}°)")

    return adjacency


def find_closed_loop(hex_results, adjacency, rotated_paths, color):
    """Ищет замкнутый маршрут заданного цвета (гамильтонов цикл)."""
    n = len(hex_results)

    # Граф: рёбра где совпадают цвета на смежных гранях
    graph = {i: [] for i in range(n)}
    for i in range(n):
        if rotated_paths[i] is None:
            continue
        for edge_i, (j, edge_j) in adjacency[i].items():
            if rotated_paths[j] is None:
                continue
            if rotated_paths[i][color][edge_i] == 1 and rotated_paths[j][color][edge_j] == 1:
                if j not in [nb for nb, _ in graph[i]]:
                    graph[i].append((j, edge_i))
                    graph[j].append((i, edge_j))

    print(f"  Граф ({color}):")
    for i in range(n):
        nbs = [(nb+1, e) for nb, e in graph[i]]
        print(f"    Tile {i+1}: {nbs}")

    # DFS поиск гамильтонова цикла
    def find_cycle(start):
        visited = {start}
        path = [start]

        def dfs(current):
            for neighbor, edge in graph[current]:
                if neighbor == start and len(path) > 2:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    if dfs(neighbor):
                        return True
                    path.pop()
                    visited.remove(neighbor)
            return False

        return path if dfs(start) else None

    # Полный цикл
    for start in range(n):
        if graph[start]:
            loop = find_cycle(start)
            if loop and len(loop) == n:
                return loop

    # Любой цикл
    for start in range(n):
        if graph[start]:
            loop = find_cycle(start)
            if loop:
                return loop

    return None


def find_route_with_alignment(image_path, class_templates, model=None,
                               target_size=128, hex_fill_ratio=0.85, debug=True):
    if model is None:
        model = SAM("sam2.1_b.pt")

    print(f"\n{'='*80}")
    print(f"ПОИСК МАРШРУТА (с выравниванием): {image_path}")
    print(f"{'='*80}")

    img_orig = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img_orig.shape[2] == 3:
        img_orig_bgra = cv2.cvtColor(img_orig, cv2.COLOR_BGR2BGRA)
    else:
        img_orig_bgra = img_orig.copy()

    # === Этап 1: SAM на оригинале ===
    print(f"\n[1] Поиск шестиугольников на оригинале (SAM)...")
    hex_masks_orig = find_hexagon_masks(image_path, model=model, debug=False)
    print(f"  Найдено: {len(hex_masks_orig)}")

    if len(hex_masks_orig) < 1:
        print("  Шестиугольники не найдены!")
        return None

    for hm in hex_masks_orig:
        ys, xs = np.where(hm['mask'] > 0)
        hm['center_x'] = xs.mean()
        hm['center_y'] = ys.mean()

    # === Этап 2: Оценка поворота ===
    if len(hex_masks_orig) >= 2:
        print(f"\n[2] Оценка поворота сетки...")
        grid_rotation = estimate_hex_grid_rotation(hex_masks_orig)
    else:
        grid_rotation = 0.0
        print(f"\n[2] Один шестиугольник — поворот 0°")

    if debug:
        img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 10))
        plt.imshow(img_rgb)
        for i, hm in enumerate(hex_masks_orig):
            plt.text(hm['center_x'], hm['center_y'], str(i + 1),
                     fontsize=12, fontweight='bold', color='white',
                     ha='center', va='center',
                     bbox=dict(boxstyle='circle,pad=0.3', facecolor='navy', alpha=0.8))
        plt.title(f"[1-2] Оригинал: {len(hex_masks_orig)} шестиугольников, "
                  f"поворот ≈ {grid_rotation:.1f}°")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    # === Этап 3: Поворот изображения и масок ===
    if abs(grid_rotation) > 0.5:
        print(f"\n[3] Поворот изображения и масок на {-grid_rotation:.1f}°...")

        h_orig, w_orig = img_orig.shape[:2]
        cx_orig, cy_orig = w_orig / 2, h_orig / 2

        rot_matrix = cv2.getRotationMatrix2D((cx_orig, cy_orig), -grid_rotation, 1.0)

        cos_a = abs(rot_matrix[0, 0])
        sin_a = abs(rot_matrix[0, 1])
        new_w = int(h_orig * sin_a + w_orig * cos_a)
        new_h = int(h_orig * cos_a + w_orig * sin_a)

        rot_matrix[0, 2] += (new_w - w_orig) / 2
        rot_matrix[1, 2] += (new_h - h_orig) / 2

        img_rotated = cv2.warpAffine(img_orig, rot_matrix, (new_w, new_h),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(0, 0, 0))

        img_rotated_bgra = cv2.warpAffine(img_orig_bgra, rot_matrix, (new_w, new_h),
                                           flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=(0, 0, 0, 0))

        rotated_hex_masks = []
        for hm in hex_masks_orig:
            mask_rotated = cv2.warpAffine(hm['mask'], rot_matrix, (new_w, new_h),
                                           flags=cv2.INTER_NEAREST,
                                           borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=0)
            ys, xs = np.where(mask_rotated > 0)
            if len(xs) == 0:
                continue
            rotated_hex_masks.append({
                **hm,
                'mask': mask_rotated,
                'area': float(mask_rotated.sum()),
                'center_x': xs.mean(),
                'center_y': ys.mean(),
            })
    else:
        print(f"\n[3] Поворот не нужен (< 0.5°)")
        img_rotated = img_orig.copy()
        img_rotated_bgra = img_orig_bgra.copy()
        rotated_hex_masks = hex_masks_orig

    if debug:
        img_rot_rgb = cv2.cvtColor(img_rotated, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        axes[0].imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Оригинал")
        axes[0].axis("off")

        axes[1].imshow(img_rot_rgb)
        for i, hm in enumerate(rotated_hex_masks):
            axes[1].text(hm['center_x'], hm['center_y'], str(i + 1),
                         fontsize=12, fontweight='bold', color='white',
                         ha='center', va='center',
                         bbox=dict(boxstyle='circle,pad=0.3', facecolor='green', alpha=0.8))
        axes[1].set_title(f"Повёрнуто на {-grid_rotation:.1f}°")
        axes[1].axis("off")

        plt.suptitle("[3] Выравнивание", fontsize=14)
        plt.tight_layout()
        plt.show()

    # === Этап 4: Классификация ===
    print(f"\n[4] Классификация...")

    rotated_hex_masks.sort(key=lambda x: x['center_x'])

    object_patches = []
    for hm in rotated_hex_masks:
        patch, center, angle, scale = extract_hex_patch_normalized(
            img_rotated_bgra, hm['mask'], target_size, hex_fill_ratio
        )
        if patch is None:
            continue

        patch_normalized, color_mask = normalize_template_colors(patch)

        object_patches.append({
            'patch_rgba': patch_normalized,
            'patch_original': patch,
            'color_mask': color_mask,
            'hex_mask': hm,
            'center': center,
            'align_angle': angle,
            'scale': scale,
        })

    print(f"  Нормализовано объектов: {len(object_patches)}")

    class_rotations = []
    for tmpl in class_templates:
        rotations = []
        for r in range(6):
            rotated = rotate_patch_60(tmpl['patch_rgba'], r)
            rotations.append(rotated)
        class_rotations.append({
            'class_name': tmpl['class_name'],
            'rotations': rotations,
        })

    hex_results = []
    for obj_idx, obj in enumerate(object_patches):
        best_score = -1
        best_class = None
        best_rotation = 0
        best_class_patch = None

        for cls in class_rotations:
            for r, rotated_template in enumerate(cls['rotations']):
                score = compare_hex_patches(obj['patch_rgba'], rotated_template)
                if score > best_score:
                    best_score = score
                    best_class = cls['class_name']
                    best_rotation = r * 60
                    best_class_patch = rotated_template

        hex_results.append({
            'mask': obj['hex_mask']['mask'],
            'class_name': best_class,
            'best_score': best_score,
            'best_rotation': best_rotation,
            'center_x': obj['hex_mask']['center_x'],
            'center_y': obj['hex_mask']['center_y'],
            'object_patch': obj['patch_rgba'],
            'object_patch_original': obj['patch_original'],
            'matched_template': best_class_patch,
            'object_scale': obj['scale'],
        })

        print(f"  Объект {obj_idx + 1}: класс={best_class}, "
              f"score={best_score:.4f}, rotation={best_rotation}°")

    if debug:
        img_rot_rgb = cv2.cvtColor(img_rotated, cv2.COLOR_BGR2RGB)
        n_obj = len(hex_results)

        plt.figure(figsize=(12, 10))
        plt.imshow(img_rot_rgb)
        for i, res in enumerate(hex_results):
            plt.text(res['center_x'], res['center_y'],
                     f"{i+1}\n{res['class_name']}\nrot={res['best_rotation']}°\n{res['best_score']:.3f}",
                     fontsize=8, fontweight='bold', color='white',
                     ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.7))
        plt.title("[4] Классификация")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        fig, axes = plt.subplots(n_obj, 4, figsize=(16, 4 * n_obj))
        if n_obj == 1:
            axes = axes.reshape(1, -1)
        for i, res in enumerate(hex_results):
            axes[i, 0].imshow(cv2.cvtColor(res['object_patch_original'], cv2.COLOR_BGRA2RGBA))
            axes[i, 0].set_title(f"Объект {i+1} оригинал", fontsize=9)
            axes[i, 0].axis("off")
            axes[i, 1].imshow(cv2.cvtColor(res['object_patch'], cv2.COLOR_BGRA2RGBA))
            axes[i, 1].set_title("Нормализован", fontsize=9)
            axes[i, 1].axis("off")
            axes[i, 2].imshow(cv2.cvtColor(res['matched_template'], cv2.COLOR_BGRA2RGBA))
            axes[i, 2].set_title(f"{res['class_name']} rot={res['best_rotation']}°", fontsize=9)
            axes[i, 2].axis("off")
            blend = (res['object_patch'].astype(np.float32) * 0.5 +
                     res['matched_template'].astype(np.float32) * 0.5).astype(np.uint8)
            axes[i, 3].imshow(cv2.cvtColor(blend, cv2.COLOR_BGRA2RGBA))
            axes[i, 3].set_title(f"Наложение {res['best_score']:.4f}", fontsize=9)
            axes[i, 3].axis("off")
        fig.suptitle("[4] Сопоставление", fontsize=13)
        plt.tight_layout()
        plt.show()

    # === Для одной фишки — только классификация, без маршрутов ===
    if len(hex_results) < 2:
        print(f"\n  Одна фишка — маршрут не строится.")
        print(f"  Результат: {hex_results[0]['class_name']} "
              f"(score={hex_results[0]['best_score']:.4f}, "
              f"rot={hex_results[0]['best_rotation']}°)")
        return {'single': hex_results[0]}

    # === Этап 5: Пути ===
    print(f"\n[5] Повёрнутые пути:")
    rotated_paths = get_rotated_paths(hex_results)

    # === Этап 6: Граф соседства ===
    print(f"\n[6] Граф соседства:")
    adjacency = build_adjacency(hex_results)

    # === Этап 7: Проверка смежных граней ===
    print(f"\n[7] Проверка смежных граней:")
    colors = ['red', 'yellow', 'blue']
    for i in range(len(hex_results)):
        for edge_i, (j, edge_j) in adjacency[i].items():
            if i < j:
                matches = []
                for color in colors:
                    if rotated_paths[i] and rotated_paths[j]:
                        if (rotated_paths[i][color][edge_i] == 1 and
                            rotated_paths[j][color][edge_j] == 1):
                            matches.append(color)
                status = ', '.join(matches) if matches else '❌ НЕТ СОВПАДЕНИЙ'
                print(f"  Tile {i+1} грань {edge_i} ↔ Tile {j+1} грань {edge_j}: {status}")

    # === Этап 8: Замкнутые маршруты ===
    print(f"\n[8] Поиск замкнутых маршрутов:")
    loops = {}
    for color in colors:
        print(f"\n  Цвет: {color}")
        loop = find_closed_loop(hex_results, adjacency, rotated_paths, color)
        if loop:
            print(f"  ✅ Замкнутый маршрут: {[idx+1 for idx in loop]}")
            loops[color] = loop
        else:
            print(f"  ❌ Замкнутый маршрут не найден")
            loops[color] = None

    # === Визуализация ===
    if debug:
        img_rot_rgb = cv2.cvtColor(img_rotated, cv2.COLOR_BGR2RGB)

        for color in colors:
            loop = loops[color]
            if loop is None:
                continue

            plt.figure(figsize=(14, 12))
            plt.imshow(img_rot_rgb)

            for step in range(len(loop)):
                i = loop[step]
                j = loop[(step + 1) % len(loop)]
                x1, y1 = hex_results[i]['center_x'], hex_results[i]['center_y']
                x2, y2 = hex_results[j]['center_x'], hex_results[j]['center_y']
                plt.annotate('', xy=(x2, y2), xytext=(x1, y1),
                             arrowprops=dict(arrowstyle='->', color=COLOR_PLT[color],
                                             lw=3, connectionstyle='arc3,rad=0.1'))

            for idx, res in enumerate(hex_results):
                order = loop.index(idx) + 1 if idx in loop else None
                label = f"{idx+1}\n{res['class_name']}"
                if order:
                    label += f"\n→{order}"
                plt.text(res['center_x'], res['center_y'], label,
                         fontsize=8, fontweight='bold', color='white',
                         ha='center', va='center',
                         bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor=COLOR_PLT[color], alpha=0.7))

            plt.title(f"Маршрут ({color}): {[idx+1 for idx in loop]}", fontsize=14)
            plt.axis("off")
            plt.tight_layout()
            plt.show()

        # Все пути
        fig, ax = plt.subplots(1, 1, figsize=(14, 12))
        ax.imshow(img_rot_rgb)

        for i, res in enumerate(hex_results):
            if rotated_paths[i] is None:
                continue
            cx, cy = res['center_x'], res['center_y']
            mask = res['mask']
            ys, xs = np.where(mask > 0)
            radius = max((xs.max() - xs.min()), (ys.max() - ys.min())) / 2

            for color_name in colors:
                edges = [e for e in range(6) if rotated_paths[i][color_name][e] == 1]
                r_off = {'red': 0.65, 'yellow': 0.75, 'blue': 0.85}

                for e_idx in edges:
                    a = np.radians(EDGE_ANGLES_DEG[e_idx])
                    d = radius * r_off[color_name]
                    ax.plot(cx + d * np.cos(a), cy - d * np.sin(a), 'o',
                            color=COLOR_PLT[color_name], markersize=8,
                            markeredgecolor='black', markeredgewidth=0.5)

                if len(edges) == 2:
                    a1 = np.radians(EDGE_ANGLES_DEG[edges[0]])
                    a2 = np.radians(EDGE_ANGLES_DEG[edges[1]])
                    d = radius * r_off[color_name]
                    ax.plot([cx + d*np.cos(a1), cx + d*np.cos(a2)],
                            [cy - d*np.sin(a1), cy - d*np.sin(a2)],
                            '-', color=COLOR_PLT[color_name], linewidth=2, alpha=0.7)

            ax.text(cx, cy, f"{i+1}", fontsize=9, fontweight='bold', color='white',
                    ha='center', va='center',
                    bbox=dict(boxstyle='circle,pad=0.2', facecolor='navy', alpha=0.8))

        ax.set_title("Все пути (выровненное изображение)", fontsize=14)
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    return loops


def estimate_hex_grid_rotation(hex_items):
    """
    Оценивает угол поворота гексагональной сетки.
    Принимает список dict с ключами center_x, center_y.
    """
    centers = np.array([(h['center_x'], h['center_y']) for h in hex_items])
    n = len(centers)

    if n < 2:
        return 0.0

    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((centers[i, 0] - centers[j, 0])**2 +
                        (centers[i, 1] - centers[j, 1])**2)
            dists.append(d)

    dists = np.array(dists)
    min_dist = dists.min()
    neighbor_threshold = min_dist * 1.4

    angles = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[j, 0] - centers[i, 0]
            dy = centers[j, 1] - centers[i, 1]
            d = np.sqrt(dx**2 + dy**2)
            if d > neighbor_threshold:
                continue
            angle = np.degrees(np.arctan2(-dy, dx))
            angles.append(angle)

    if not angles:
        return 0.0

    offsets = []
    for a in angles:
        remainder = a % 60
        if remainder > 30:
            offset = remainder - 60
        else:
            offset = remainder
        offsets.append(offset)

    grid_rotation = np.median(offsets)

    print(f"  Углы к соседям: {[f'{a:.1f}°' for a in angles]}")
    print(f"  Отклонения от кратных 60°: {[f'{o:.1f}°' for o in offsets]}")
    print(f"  Оценка поворота сетки: {grid_rotation:.1f}°")

    return grid_rotation