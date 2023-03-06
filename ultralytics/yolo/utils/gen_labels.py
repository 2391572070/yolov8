from tqdm import tqdm
import os
import numpy as np
from pathlib import Path
from ultralytics.yolo.utils.checks import check_requirements
from ultralytics.yolo.utils.ops import xyxy2xywhn
check_requirements(('pycocotools>=2.0',))
from pycocotools.coco import COCO


def gen_labels(names, out_dir, data_root, out_images_file, ann_file_path, image_dir, skip_crowd=True, skip_reflected=False, object_min_area=None, skips=None):
    print('=' * 80)
    print(ann_file_path, image_dir)
    coco = COCO(data_root / ann_file_path)
    ann_dir = out_dir / 'labels' / image_dir
    if not ann_dir.exists():
        ann_dir.mkdir(parents=True, exist_ok=True)

    if skips is not None:
        skips = set(skips)
        if len(skips) < 1:
            skips = None

    class_count = len(names)
    image_name_set = set()
    for cid, cat in names.items():
        if skips is not None:
            if cat in skips:
                continue
        if object_min_area is not None:
            min_area = object_min_area.get(cat, 0)
        else:
            min_area = 0
        catIds = coco.getCatIds(catNms=[cat])
        if len(catIds) != 1:
            continue
        imgIds = coco.getImgIds(catIds=catIds)
        for im in tqdm(coco.loadImgs(imgIds), desc=f'Class {cid + 1}/{class_count} {cat}'):
            width, height = im["width"], im["height"]
            path = Path(im["file_name"])  # image filename
            try:
                ann_path = ann_dir / path.with_suffix('.txt')
                if not ann_path.parent.exists():
                    ann_path.parent.mkdir(parents=True, exist_ok=True)
                annIds = coco.getAnnIds(imgIds=im["id"], catIds=catIds, iscrowd=None)
                label_info_set = set()
                label_info_list = []
                for a in coco.loadAnns(annIds):
                    if a.get('ignore', False):
                        continue
                    if skip_crowd and a.get('iscrowd', False):
                        continue
                    if skip_reflected and a.get('isreflected', False):
                        continue
                    x, y, w, h = a['bbox']  # bounding box in xywh (xy top-left corner)
                    if a['area'] <= 0 or w < 1 or h < 1:
                        continue
                    if x < 0:
                        x = 0
                    if y < 0:
                        y = 0
                    x2 = x + w
                    y2 = y + h
                    if x2 > width:
                        x2 = width
                    if y2 > height:
                        y2 = height
                    w = x2 - x
                    h = y2 - y
                    if w < 1 or h < 1 or (w * h < min_area):
                        continue
                    # x, y = x + w / 2, y + h / 2  # xy to center
                    # x, y, w, h = x/width, y/height, w/width, h/height
                    xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)[None]  # pixels(1,4)
                    x, y, w, h = xyxy2xywhn(xyxy, w=width, h=height, clip=True)[0]  # normalized and clipped
                    label_info = f"{cid} {x:.5f} {y:.5f} {w:.5f} {h:.5f}\n"
                    if label_info in label_info_set:
                        continue
                    label_info_set.add(label_info)
                    label_info_list.append(label_info)
                if len(label_info_list) > 0:
                    with open(ann_path, 'a') as file:
                        for label_info in label_info_list:
                            file.write(label_info)
                    image_name = str(path)
                    if image_name not in image_name_set:
                        out_images_file.write(f"./{os.path.normpath(os.path.join('images', image_dir, image_name))}\n")
                        image_name_set.add(image_name)
            except Exception as e:
                print(e)


def gen_backgrouds(out_dir, data_root, out_images_file, ann_file_path, image_dir):
    print('=' * 80)
    print(ann_file_path, image_dir)
    ann_dir = out_dir / 'labels' / image_dir
    if not ann_dir.exists():
        ann_dir.mkdir(parents=True, exist_ok=True)

    image_name_set = set()
    with open(data_root / ann_file_path, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            line = line.strip()
            if line == '':
                continue
            try:
                path = Path(line.split()[0])
                ann_path = ann_dir / path.with_suffix('.txt')
                if not ann_path.parent.exists():
                    ann_path.parent.mkdir(parents=True, exist_ok=True)
                with open(ann_path, 'w', encoding='utf-8') as _:
                    pass
                image_name = str(path)
                if image_name not in image_name_set:
                    out_images_file.write(f"./{os.path.normpath(os.path.join('images', image_dir, image_name))}\n")
                    image_name_set.add(image_name)
            except Exception as e:
                print(e)
