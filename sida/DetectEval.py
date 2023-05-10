#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author:  yerunyuan
@contact: yerunyuan@163.com
"""
import sys
sys.path.insert(0, '.')

import argparse
import os
import json
import itertools
from terminaltables import AsciiTable
import numpy as np

from pycocotools.coco import COCO

# from pycocotools.cocoeval import COCOeval
try:
    from yoloxtools import COCOeval_opt as COCOeval
except ImportError:
    from pycocotools.cocoeval import COCOeval
    print("Use standard COCOeval.")


def parse_args():
    parser = argparse.ArgumentParser(description='Detection Eval')
    parser.add_argument('--bbox_list_file', type=str, required=True, help='detection bbox list file')
    parser.add_argument("--ann_json_file", type=str, required=True, help='annotations json file')
    parser.add_argument("--classwise", action="store_true", help="Whether to evaluating the AP for each class")
    parser.add_argument("--label_offset", type=int, default=0, help='label offset')
    parser.add_argument('--image_dir', type=str, default=None, help='path to image directory')
    parser.add_argument("--display", action="store_true", help="display")
    parser.add_argument("--score_thr", type=float, default=0, help='display score thr')
    parser.add_argument("--bbox_size_unit", default=1, help="0 is pixel unit, 1 is 0~1", type=int)
    args = parser.parse_args()
    return args


def os_system(cmd_str):
    if sys.platform.startswith('linux'):
        os.system(cmd_str)


def main():
    args = parse_args()

    classwise = args.classwise
    label_offset = args.label_offset
    bbox_size_unit = args.bbox_size_unit

    pred_anns = []
    id_pred_anns = {}
    anno = None
    ext = os.path.splitext(args.bbox_list_file)[-1].lower()
    if ext == '.json':
        anno = COCO(args.ann_json_file)  # init annotations api
        base_ids = {}
        id_filenames = {}
        for image_id, image_info in anno.imgs.items():
            file_name = image_info['file_name']
            base_name = os.path.splitext(os.path.basename(file_name))[0]
            base_ids[base_name] = image_id
            id_filenames[image_id] = file_name

        with open(args.bbox_list_file, 'r', encoding='utf-8') as file:
            json_info = json.load(file)
            for info in json_info:
                image_id = info['image_id']
                image_id = base_ids.get(image_id, None)
                if image_id is None:
                    continue
                info['image_id'] = image_id
                if 0 != label_offset:
                    info['category_id'] += label_offset
                pred_anns.append(info)
                _bbox_info = [info['category_id'], info['bbox'], info['score']]
                _pred_anns = id_pred_anns.get(image_id, None)
                if _pred_anns is None:
                    id_pred_anns[image_id] = [id_filenames[image_id], [_bbox_info]]
                else:
                    _pred_anns[1].append(_bbox_info)

    if args.display:
        import cv2
        image_dir = args.image_dir
        if image_dir is None:
            return

        score_thr = args.score_thr

        win_name = 'Display'
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)

        if len(id_pred_anns) > 0:
            id_pred_anns = sorted(id_pred_anns.items())
            for _, pred_ann in id_pred_anns:
                file_name = pred_ann[0]
                bbox_info = pred_ann[1]
                image_path = os.path.join(image_dir, file_name)
                if not os.path.exists(image_path):
                    print(image_path, 'not exists.')
                    continue
                image = cv2.imread(image_path)

                height, width = image.shape[:2]

                for label, bbox, score in bbox_info:
                    if score < score_thr:
                        continue
                    x1, y1, x2, y2 = bbox[0], bbox[1], (bbox[0]+bbox[2]), (bbox[1]+bbox[3])
                    x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
                    display_str = '{} {:.02f}'.format(label, score)
                    cv2.putText(image, display_str, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), thickness=2)
                cv2.imshow(win_name, image)
                key = cv2.waitKey()
                if key == 27:
                    break
        else:
            with open(args.bbox_list_file, 'r') as file:
                for line in file.readlines():
                    lines = line.strip().split()
                    if len(lines) < 2:
                        continue
                    file_name = lines[0]
                    bbox_info = lines[1].split(';')
                    image_path = os.path.join(image_dir, file_name)
                    if not os.path.exists(image_path):
                        print(image_path, 'not exists.')
                        continue
                    image = cv2.imread(image_path)

                    height, width = image.shape[:2]

                    for bbox in bbox_info:
                        bbox = bbox.split(',')
                        score = float(bbox[5])
                        if score < score_thr:
                            continue
                        label = int(bbox[0]) + label_offset
                        x1, y1, x2, y2 = float(bbox[1]), float(bbox[2]), float(bbox[3]), float(bbox[4])
                        if 1 == bbox_size_unit:
                            x1, y1, x2, y2 = round(x1*width), round(y1*height), round(x2*width), round(y2*height)
                        else:
                            x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
                        display_str = '{} {:.02f}'.format(label, score)
                        cv2.putText(image, display_str, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), thickness=2)
                    cv2.imshow(win_name, image)
                    key = cv2.waitKey()
                    if key == 27:
                        break
        return

    if len(pred_anns) < 1:
        file_bboxes = {}
        with open(args.bbox_list_file, 'r') as file:
            for line in file.readlines():
                lines = line.strip().split()
                if len(lines) < 2:
                    continue
                file_name = lines[0]
                bbox_info = lines[1].split(';')
                bboxes = []
                for bbox in bbox_info:
                    bbox = bbox.split(',')
                    bboxes.append([int(bbox[0])+label_offset, float(bbox[1]), float(bbox[2]), float(bbox[3]), float(bbox[4]), float(bbox[5])])
                if len(bboxes) > 0:
                    file_bboxes[file_name] = bboxes

    try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
        if anno is None:
            anno = COCO(args.ann_json_file)  # init annotations api

        if len(pred_anns) < 1:
            for image_id, image_info in anno.imgs.items():
                file_name = image_info['file_name']
                bboxes = file_bboxes.get(file_name, None)
                if bboxes is None:
                    continue
                if 1 == bbox_size_unit:
                    width = image_info['width']
                    height = image_info['height']
                    for bbox in bboxes:
                        x1 = bbox[1] * width
                        y1 = bbox[2] * height
                        x2 = bbox[3] * width
                        y2 = bbox[4] * height
                        pred_anns.append({'image_id': image_id,
                                      'category_id': bbox[0],
                                      'bbox': [x1, y1, (x2-x1), (y2-y1)],
                                      'score': bbox[5]})
                else:
                    for bbox in bboxes:
                        x1, y1, x2, y2 = bbox[1:5]
                        pred_anns.append({'image_id': image_id,
                                      'category_id': bbox[0],
                                      'bbox': [x1, y1, (x2-x1), (y2-y1)],
                                      'score': bbox[5]})

        pred = anno.loadRes(pred_anns)  # init predictions api
        eval = COCOeval(anno, pred, 'bbox')
        # if is_coco:
        #     eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
        eval.evaluate()
        eval.accumulate()
        eval.summarize()
        map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)

        if classwise:  # Compute per-category AP
            # Compute per-category AP
            # from https://github.com/facebookresearch/detectron2/
            precisions = eval.eval['precision']
            # precision: (iou, recall, cls, area range, max dets)
            cat_ids = eval.params.catIds
            assert len(cat_ids) == precisions.shape[2]

            results_per_category = []
            for idx, catId in enumerate(cat_ids):
                # area range index 0: all area ranges
                # max dets index -1: typically 100 per image
                nm = anno.loadCats([catId])[0]
                precision = precisions[:, :, idx, 0, -1]
                precision = precision[precision > -1]
                if precision.size:
                    _ap = np.mean(precision)
                else:
                    _ap = float('nan')
                results_per_category.append(
                    (f'{nm["name"]}', f'{float(_ap):0.3f}'))

            num_columns = min(6, len(results_per_category) * 2)
            results_flatten = list(
                itertools.chain(*results_per_category))
            headers = ['category', 'AP'] * (num_columns // 2)
            results_2d = itertools.zip_longest(*[
                results_flatten[i::num_columns]
                for i in range(num_columns)
            ])
            table_data = [headers]
            table_data += [result for result in results_2d]
            table = AsciiTable(table_data)
            print('\n' + table.table)
    except Exception as e:
        print(f'pycocotools unable to run: {e}')

    print('Finish!')


if __name__ == '__main__':
    main()

