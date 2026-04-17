import gradio as gr
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from modules import *


def fig_to_pil(fig):
    """Конвертирует matplotlib figure в PIL Image."""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    pil_img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    return pil_img


def process_image(input_image):
    """
    Основная функция Gradio.
    Принимает изображение, возвращает галерею промежуточных этапов и текстовый отчёт.
    """
    if input_image is None:
        return [], "Загрузите изображение!"

    gallery = []
    report_lines = []

    # Сохраняем временный файл для SAM
    temp_path = "/tmp/gradio_input.png"
    if isinstance(input_image, np.ndarray):
        cv2.imwrite(temp_path, cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR))
    else:
        input_image.save(temp_path)

    img_orig = cv2.imread(temp_path, cv2.IMREAD_UNCHANGED)
    if img_orig is None:
        return [], "Не удалось загрузить изображение!"

    if len(img_orig.shape) == 2:
        img_orig = cv2.cvtColor(img_orig, cv2.COLOR_GRAY2BGR)
    if img_orig.shape[2] == 3:
        img_orig_bgra = cv2.cvtColor(img_orig, cv2.COLOR_BGR2BGRA)
    else:
        img_orig_bgra = img_orig.copy()

    img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

    # ============================================================
    # Этап 1: SAM — поиск шестиугольников
    # ============================================================
    report_lines.append("=" * 60)
    report_lines.append("ЭТАП 1: Поиск шестиугольников (SAM)")
    report_lines.append("=" * 60)

    hex_masks_orig = find_hexagon_masks(temp_path, model=model, debug=False)
    report_lines.append(f"Найдено шестиугольников: {len(hex_masks_orig)}")

    if len(hex_masks_orig) == 0:
        return [], "Шестиугольники не найдены на изображении!"

    # Вычисляем центры
    for hm in hex_masks_orig:
        ys, xs = np.where(hm['mask'] > 0)
        hm['center_x'] = xs.mean()
        hm['center_y'] = ys.mean()

    # Визуализация SAM
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(img_rgb)
    cmap = plt.cm.get_cmap("tab10")
    for i, hm in enumerate(hex_masks_orig):
        color = cmap(i % 10)
        mask_overlay = np.zeros((*img_rgb.shape[:2], 4))
        mask_overlay[hm['mask'] > 0] = [color[0], color[1], color[2], 0.4]
        ax.imshow(mask_overlay)
        ax.text(hm['center_x'], hm['center_y'], str(i + 1),
                fontsize=14, fontweight='bold', color='white',
                ha='center', va='center',
                bbox=dict(boxstyle='circle,pad=0.3', facecolor=color, alpha=0.8))
    ax.set_title(f"Этап 1: SAM — найдено {len(hex_masks_orig)} шестиугольников")
    ax.axis("off")
    gallery.append(fig_to_pil(fig))

    # ============================================================
    # Этап 2: Оценка и коррекция поворота
    # ============================================================
    report_lines.append("")
    report_lines.append("=" * 60)
    report_lines.append("ЭТАП 2: Выравнивание сетки")
    report_lines.append("=" * 60)

    if len(hex_masks_orig) >= 2:
        grid_rotation = estimate_hex_grid_rotation(hex_masks_orig)
    else:
        grid_rotation = 0.0

    report_lines.append(f"Поворот сетки: {grid_rotation:.1f}°")

    if abs(grid_rotation) > 0.5:
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
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        img_rotated_bgra = cv2.warpAffine(img_orig_bgra, rot_matrix, (new_w, new_h),
                                           flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        rotated_hex_masks = []
        for hm in hex_masks_orig:
            mask_r = cv2.warpAffine(hm['mask'], rot_matrix, (new_w, new_h),
                                     flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            ys, xs = np.where(mask_r > 0)
            if len(xs) == 0:
                continue
            rotated_hex_masks.append({
                **hm, 'mask': mask_r, 'area': float(mask_r.sum()),
                'center_x': xs.mean(), 'center_y': ys.mean(),
            })
    else:
        img_rotated = img_orig.copy()
        img_rotated_bgra = img_orig_bgra.copy()
        rotated_hex_masks = hex_masks_orig

    img_rot_rgb = cv2.cvtColor(img_rotated, cv2.COLOR_BGR2RGB)

    # Визуализация выравнивания
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(img_rgb)
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
    fig.suptitle("Этап 2: Выравнивание сетки")
    gallery.append(fig_to_pil(fig))

    # ============================================================
    # Этап 3: Нормализация цветов и классификация
    # ============================================================
    report_lines.append("")
    report_lines.append("=" * 60)
    report_lines.append("ЭТАП 3: Нормализация цветов и классификация")
    report_lines.append("=" * 60)

    rotated_hex_masks.sort(key=lambda x: x['center_x'])
    target_size = 128
    hex_fill_ratio = 0.85

    object_patches = []
    for hm in rotated_hex_masks:
        patch, center, angle, scale = extract_hex_patch_normalized(
            img_rotated_bgra, hm['mask'], target_size, hex_fill_ratio
        )
        if patch is None:
            continue
        patch_normalized, color_mask = normalize_template_colors(patch)
        object_patches.append({
            'patch_rgba': patch_normalized, 'patch_original': patch,
            'color_mask': color_mask, 'hex_mask': hm,
            'center': center, 'align_angle': angle, 'scale': scale,
        })

    # Визуализация нормализации цветов
    n_obj = len(object_patches)
    if n_obj > 0:
        fig, axes = plt.subplots(n_obj, 3, figsize=(12, 4 * n_obj))
        if n_obj == 1:
            axes = axes.reshape(1, -1)
        for i, op in enumerate(object_patches):
            axes[i, 0].imshow(cv2.cvtColor(op['patch_original'], cv2.COLOR_BGRA2RGBA))
            axes[i, 0].set_title(f"Объект {i+1} оригинал", fontsize=9)
            axes[i, 0].axis("off")

            axes[i, 1].imshow(cv2.cvtColor(
                gray_world_normalize_bgra(op['patch_original']), cv2.COLOR_BGRA2RGBA))
            axes[i, 1].set_title("Gray World", fontsize=9)
            axes[i, 1].axis("off")

            axes[i, 2].imshow(cv2.cvtColor(op['patch_rgba'], cv2.COLOR_BGRA2RGBA))
            axes[i, 2].set_title("Нормализован", fontsize=9)
            axes[i, 2].axis("off")
        fig.suptitle("Этап 3: Нормализация цветов")
        gallery.append(fig_to_pil(fig))

    # Классификация
    class_rotations = []
    for tmpl in class_templates:
        rotations = [rotate_patch_60(tmpl['patch_rgba'], r) for r in range(6)]
        class_rotations.append({'class_name': tmpl['class_name'], 'rotations': rotations})

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
            'class_name': best_class, 'best_score': best_score,
            'best_rotation': best_rotation,
            'center_x': obj['hex_mask']['center_x'],
            'center_y': obj['hex_mask']['center_y'],
            'object_patch': obj['patch_rgba'],
            'object_patch_original': obj['patch_original'],
            'matched_template': best_class_patch,
            'object_scale': obj['scale'],
        })

        # Извлекаем номер класса
        class_num = ''.join(filter(str.isdigit, best_class)) if best_class else '?'
        report_lines.append(f"Объект {obj_idx+1}: класс {class_num} ({best_class}), "
                           f"score={best_score:.4f}, поворот={best_rotation}°")

    # ============================================================
    # Этап 4: Сопоставление шаблонов
    # ============================================================
    report_lines.append("")
    report_lines.append("=" * 60)
    report_lines.append("ЭТАП 4: Сопоставление с шаблонами")
    report_lines.append("=" * 60)

    if n_obj > 0:
        fig, axes = plt.subplots(n_obj, 4, figsize=(16, 4 * n_obj))
        if n_obj == 1:
            axes = axes.reshape(1, -1)
        for i, res in enumerate(hex_results):
            axes[i, 0].imshow(cv2.cvtColor(res['object_patch_original'], cv2.COLOR_BGRA2RGBA))
            axes[i, 0].set_title(f"Объект {i+1}", fontsize=9)
            axes[i, 0].axis("off")

            axes[i, 1].imshow(cv2.cvtColor(res['object_patch'], cv2.COLOR_BGRA2RGBA))
            axes[i, 1].set_title("Нормализован", fontsize=9)
            axes[i, 1].axis("off")

            axes[i, 2].imshow(cv2.cvtColor(res['matched_template'], cv2.COLOR_BGRA2RGBA))
            class_num = ''.join(filter(str.isdigit, res['class_name'])) if res['class_name'] else '?'
            axes[i, 2].set_title(f"Класс {class_num}\nrot={res['best_rotation']}°", fontsize=9)
            axes[i, 2].axis("off")

            blend = (res['object_patch'].astype(np.float32) * 0.5 +
                     res['matched_template'].astype(np.float32) * 0.5).astype(np.uint8)
            axes[i, 3].imshow(cv2.cvtColor(blend, cv2.COLOR_BGRA2RGBA))
            axes[i, 3].set_title(f"Score: {res['best_score']:.4f}", fontsize=9)
            axes[i, 3].axis("off")

            report_lines.append(f"  Объект {i+1} → Класс {class_num} "
                               f"(score={res['best_score']:.4f})")
        fig.suptitle("Этап 4: Сопоставление с шаблонами")
        gallery.append(fig_to_pil(fig))

    # Классификация на изображении
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.imshow(img_rot_rgb)
    for i, res in enumerate(hex_results):
        class_num = ''.join(filter(str.isdigit, res['class_name'])) if res['class_name'] else '?'
        ax.text(res['center_x'], res['center_y'],
                f"{i+1}\nКласс {class_num}\n{res['best_score']:.3f}",
                fontsize=9, fontweight='bold', color='white',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.7))
    ax.set_title("Результат классификации")
    ax.axis("off")
    gallery.append(fig_to_pil(fig))

    # ============================================================
    # Этап 5: Маршруты (если >= 2 фишек)
    # ============================================================
    if len(hex_results) >= 2:
        report_lines.append("")
        report_lines.append("=" * 60)
        report_lines.append("ЭТАП 5: Построение маршрутов")
        report_lines.append("=" * 60)

        rotated_paths = get_rotated_paths(hex_results)
        adjacency = build_adjacency(hex_results)

        colors = ['red', 'yellow', 'blue']
        loops = {}

        # Проверка граней
        for i in range(len(hex_results)):
            for edge_i, (j, edge_j) in adjacency[i].items():
                if i < j:
                    matches = []
                    for color in colors:
                        if rotated_paths[i] and rotated_paths[j]:
                            if (rotated_paths[i][color][edge_i] == 1 and
                                rotated_paths[j][color][edge_j] == 1):
                                matches.append(color)
                    status = ', '.join(matches) if matches else 'нет'
                    report_lines.append(f"  Tile {i+1}↔{j+1}: {status}")

        # Поиск маршрутов
        for color in colors:
            loop = find_closed_loop(hex_results, adjacency, rotated_paths, color)
            if loop:
                loops[color] = loop
                route_str = ' → '.join([str(idx+1) for idx in loop])
                report_lines.append(f"  ✅ {color}: {route_str}")
            else:
                loops[color] = None
                report_lines.append(f"  ❌ {color}: маршрут не найден")

        # Визуализация путей на карте
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

            class_num = ''.join(filter(str.isdigit, res['class_name'])) if res['class_name'] else '?'
            ax.text(cx, cy, f"{i+1}\n({class_num})", fontsize=9, fontweight='bold',
                    color='white', ha='center', va='center',
                    bbox=dict(boxstyle='circle,pad=0.2', facecolor='navy', alpha=0.8))

        ax.set_title("Этап 5: Все пути")
        ax.axis("off")
        gallery.append(fig_to_pil(fig))

        # Визуализация каждого маршрута
        for color in colors:
            loop = loops[color]
            if loop is None:
                continue

            fig, ax = plt.subplots(1, 1, figsize=(14, 12))
            ax.imshow(img_rot_rgb)

            for step in range(len(loop)):
                i = loop[step]
                j = loop[(step + 1) % len(loop)]
                x1, y1 = hex_results[i]['center_x'], hex_results[i]['center_y']
                x2, y2 = hex_results[j]['center_x'], hex_results[j]['center_y']
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                            arrowprops=dict(arrowstyle='->', color=COLOR_PLT[color],
                                            lw=3, connectionstyle='arc3,rad=0.1'))

            for idx, res in enumerate(hex_results):
                order = loop.index(idx) + 1 if idx in loop else None
                class_num = ''.join(filter(str.isdigit, res['class_name'])) if res['class_name'] else '?'
                label = f"{idx+1} (кл.{class_num})"
                if order:
                    label += f"\n→{order}"
                ax.text(res['center_x'], res['center_y'], label,
                        fontsize=8, fontweight='bold', color='white',
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor=COLOR_PLT[color], alpha=0.7))

            route_str = ' → '.join([str(idx+1) for idx in loop])
            ax.set_title(f"Маршрут {color}: {route_str}", fontsize=14)
            ax.axis("off")
            gallery.append(fig_to_pil(fig))

    else:
        report_lines.append("")
        report_lines.append("Одна фишка — маршрут не строится.")

    # Итоговый отчёт
    report_lines.append("")
    report_lines.append("=" * 60)
    report_lines.append("ИТОГ")
    report_lines.append("=" * 60)
    for i, res in enumerate(hex_results):
        class_num = ''.join(filter(str.isdigit, res['class_name'])) if res['class_name'] else '?'
        report_lines.append(f"  Фишка {i+1}: Класс {class_num} "
                           f"(поворот {res['best_rotation']}°, score {res['best_score']:.4f})")

    report = "\n".join(report_lines)
    return gallery, report


# =============================================
# Инициализация модели и шаблонов
# =============================================

model = SAM("sam2.1_b.pt")

print("Загрузка шаблонов:")
class_templates = load_class_templates("hexagons", target_size=128, hex_fill_ratio=0.85, debug=False)
print(f"Загружено: {len(class_templates)}")

# =============================================
# Gradio интерфейс
# =============================================

demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy", label="Загрузите фото с фишками Тантрикс"),
    outputs=[
        gr.Gallery(label="Этапы обработки", columns=1, height="auto"),
        gr.Textbox(label="Отчёт", lines=30),
    ],
    title="🔷 Tantrix — Распознавание фишек и построение маршрутов",
    description=(
        "Загрузите фотографию с фишками Тантрикс.\n"
        "Система определит класс каждой фишки (1–10), "
        "найдёт замкнутые маршруты по каждому цвету "
        "и покажет все промежуточные этапы обработки."
    ),
)

demo.launch(share=True)