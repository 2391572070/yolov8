#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author:  yerunyuan
@contact: yerunyuan@163.com
"""
import sys
sys.path.insert(0, '.')

import argparse
import math
import os
import json
import numpy as np
import datetime
import time
import copy

from pycocotools.coco import COCO

from pycocotools.cocoeval import COCOeval


def parse_args():
    parser = argparse.ArgumentParser(description='Sida Detect Eval')
    parser.add_argument('--bbox_list_file', type=str, required=True, help='detection bbox list file')
    parser.add_argument("--ann_json_file", type=str, required=True, help='annotations json file')
    parser.add_argument("--classwise", action="store_true", help="Whether to evaluating the AP for each class")
    parser.add_argument("--label_offset", type=int, default=0, help='label offset')
    parser.add_argument("--src_labels", default=None, help="Space separated value of the src labels", nargs='*', type=int)
    parser.add_argument("--dst_labels", default=None, help="Space separated value of the dst labels", nargs='*', type=int)
    parser.add_argument("--score_thr_start", type=float, default=0.10, help='the start of score thr')
    parser.add_argument("--score_thr_end", type=float, default=0.60, help='the end of score thr')
    parser.add_argument("--score_thr_step", type=float, default=0.05, help='the step of score thr')
    parser.add_argument("--iou_thrs", default=[0.5, 0.75], help="Space separated value of the dst labels", nargs='*', type=float)
    parser.add_argument('--image_dir', type=str, default=None, help='path to image directory')
    parser.add_argument("--display", action="store_true", help="display")
    parser.add_argument("--score_thr", type=float, default=0, help='display score thr')
    parser.add_argument("--bbox_size_unit", default=1, help="0 is pixel unit, 1 is 0~1", type=int)
    args = parser.parse_args()
    return args


def os_system(cmd_str):
    if sys.platform.startswith('linux'):
        os.system(cmd_str)


class SidaCOCOeval(COCOeval):
    def __init__(self, cocoGt=None, cocoDt=None, iouType='bbox', scoreThrs=None, iouThrs=None, classwise=False):
        super(SidaCOCOeval, self).__init__(cocoGt=cocoGt, cocoDt=cocoDt, iouType=iouType)
        if scoreThrs is None:
            self.scoreThrs = SidaCOCOeval.cal_range(0.25, 0.55, 0.01)
        else:
            self.scoreThrs = np.array(scoreThrs, dtype=np.float32)

        if iouThrs is None:
            self.iouThrs = SidaCOCOeval.cal_range(0.5, 0.95, .05)
        else:
            self.iouThrs = np.array(iouThrs, dtype=np.float32)
        self.classwise = classwise

    @staticmethod
    def cal_range(start, end, step):
        return np.linspace(start, end, int(np.round((end - start) / step)) + 1, endpoint=True, dtype=np.float32)

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params=p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = self.computeIoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks
        self.ious = {(imgId, catId): computeIoU(imgId, catId) \
                        for imgId in p.imgIds
                        for catId in catIds}

        scoreThrs = self.scoreThrs
        imgIds = p.imgIds
        evaluateImg = self.evaluateImg
        self.evalImgs = [evaluateImg(imgId, catId, scoreThr)
                 for catId in catIds
                 for scoreThr in scoreThrs
                 for imgId in imgIds
             ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    def evaluateImg(self, imgId, catId, scoreThr):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            if g['ignore']:
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        # dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dtScores = np.array([d['score'] for d in dt], dtype=np.float32)
        ndtScores = -dtScores
        dtind = np.argsort(ndtScores, kind='mergesort')
        if scoreThr > 0.0:
            ndtScores = ndtScores[dtind]
            nscoreThr = -scoreThr
            last_ind = np.searchsorted(ndtScores, nscoreThr, side='right')
            if last_ind < ndtScores.size:
                dtind = dtind[:last_ind]
        dt = [dt[i] for i in dtind]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(self.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))
        if not len(ious)==0:
            for tind, t in enumerate(self.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind,gind]>0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind,gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou=ious[dind,gind]
                        m=gind
                    # if match made store id of match for both dt and gt
                    if m ==-1:
                        continue
                    dtIg[tind,dind] = gtIg[m]
                    dtm[tind,dind]  = gt[m]['id']
                    gtm[tind,m]     = d['id']

        # store results for given image and category
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'scoreThr':     scoreThr,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
            }

    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(self.iouThrs)
        K           = len(p.catIds) if p.useCats else 1
        A           = len(self.scoreThrs)
        tpCount = np.zeros((T, K, A), dtype=np.int32)
        fpCount = np.zeros((T, K, A), dtype=np.int32)
        fnCount = np.zeros((T, K, A), dtype=np.int32)
        dtCount = np.zeros((T, K, A), dtype=np.int32)
        gtCount = np.zeros((K,), dtype=np.int32)

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        a_list = [n for n in range(A)]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0*A0*I0
            for a, a0 in enumerate(a_list):
                Na = a0*I0
                Ni = Nk + Na
                E = [self.evalImgs[Ni + i] for i in i_list]
                E = [e for e in E if not e is None]
                if len(E) == 0:
                    continue
                dtScores = np.concatenate([e['dtScores'] for e in E])

                # different sorting method generates slightly different results.
                # mergesort is used to be consistent as Matlab implementation.
                inds = np.argsort(-dtScores, kind='mergesort')
                # dtScoresSorted = dtScores[inds]

                dtm  = np.concatenate([e['dtMatches'] for e in E], axis=1)[:,inds]
                dtIg = np.concatenate([e['dtIgnore'] for e in E], axis=1)[:,inds]
                gtIg = np.concatenate([e['gtIgnore'] for e in E])
                npig = np.count_nonzero(gtIg==0)
                gtCount[k] = npig

                tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
                fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

                tp = np.sum(tps, axis=1).astype(dtype=np.float32)
                fp = np.sum(fps, axis=1).astype(dtype=np.float32)

                tp = tp.tolist()
                fp = fp.tolist()

                for t, (_tp, _fp) in enumerate(zip(tp, fp)):
                    tpCount[t, k, a] = _tp
                    fpCount[t, k, a] = _fp
                    fnCount[t, k, a] = npig - _tp
                    dtCount[t, k, a] = _tp + _fp

        self.eval = {
            'params': p,
            'counts': [T, A, K],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'tpCount': tpCount,
            'fpCount': fpCount,
            'fnCount': fnCount,
            'dtCount': dtCount,
            'gtCount': gtCount,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))

    # 精确率和召回率同样重要
    def F1(self, precision, recall):
        return 2 * (precision * recall) / (precision + recall)

    # b大于1，召回率的权重高于精确率，b小于1精确率的权重高于召回率，常用F2、F0.5
    def Fb(self, precision, recall, b):
        b2 = b ** 2
        return (1 + b2) * (precision * recall) / (b2 * precision + recall)

    def Dist(self, precision, recall):
        return math.sqrt((precision * precision + recall * recall)*0.5)

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        iouThrCount = len(self.iouThrs)
        def _summarize(iouThr):
            print()

            t = np.where(iouThr == self.iouThrs)[0][0]
            tpCount = np.array(self.eval['tpCount'][t], dtype=np.int32)
            fpCount = np.array(self.eval['fpCount'][t], dtype=np.int32)
            fnCount = np.array(self.eval['fnCount'][t], dtype=np.int32)
            dtCount = np.array(self.eval['dtCount'][t], dtype=np.int32)
            gtCount = np.array(self.eval['gtCount'], dtype=np.int32)

            _tpCount = np.sum(tpCount, axis=0)
            _fpCount = np.sum(fpCount, axis=0)
            _fnCount = np.sum(fnCount, axis=0)
            _dtCount = np.sum(dtCount, axis=0)
            _gtCount = np.sum(gtCount)

            nan = float('nan')
            max_f1 = 0
            max_str = None
            iStr = 'IoU={:0.3f} Score={:0.3f} @ [ F1={:0.3f}, AP={:0.3f}, AR={:0.3f}, TP={}, FP={}, FN={}, FP+FN={}, DT={}, GT={} ]'
            for a, scoreThr in enumerate(self.scoreThrs):
                tp = _tpCount[a]
                fp = _fpCount[a]
                fn = _fnCount[a]
                dt = _dtCount[a]
                gt = _gtCount
                if dt != 0:
                    pr = tp / dt
                else:
                    pr = nan
                if gt != 0:
                    rc = tp / gt
                else:
                    rc = nan
                f1 = self.F1(pr, rc)
                # f1 = self.Dist(pr, rc)
                sumStr = iStr.format(iouThr, scoreThr, f1, pr, rc, tp, fp, fn, fp+fn, dt, gt)
                if max_f1 < f1:
                    max_f1 = f1
                    max_str = sumStr
                print(sumStr)
            # print('Best F1 Score:')
            # print(max_str)

            if self.classwise:
                cat_ids = self.params.catIds
                print('=' * 80)
                for idx, catId in enumerate(cat_ids):
                    max_f1 = 0
                    max_str = None
                    nm = self.cocoGt.loadCats([catId])[0]["name"]
                    gt = gtCount[idx]
                    for a, scoreThr in enumerate(self.scoreThrs):
                        tp = tpCount[idx, a]
                        fp = fpCount[idx, a]
                        fn = fnCount[idx, a]
                        dt = dtCount[idx, a]
                        if dt != 0:
                            pr = tp / dt
                        else:
                            pr = nan
                        if gt != 0:
                            rc = tp / gt
                        else:
                            rc = nan
                        f1 = self.F1(pr, rc)
                        # f1 = self.Dist(pr, rc)
                        sumStr = iStr.format(iouThr, scoreThr, f1, pr, rc, tp, fp, fn, fp+fn, dt, gt)
                        if max_f1 < f1:
                            max_f1 = f1
                            max_str = sumStr
                        print(nm, sumStr)
            return 0

        def _summarizeDets():
            stats = np.zeros((iouThrCount,))
            for i, iouThr in enumerate(self.iouThrs):
                stats[i] = _summarize(iouThr=iouThr)
            return stats
        def _summarizeKps():
            stats = np.zeros((iouThrCount,))
            for i, iouThr in enumerate(self.iouThrs):
                stats[i] = _summarize(iouThr=iouThr)
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()

    def __str__(self):
        self.summarize()


def main():
    args = parse_args()

    label_offset = args.label_offset
    bbox_size_unit = args.bbox_size_unit

    label_map = None
    if (args.src_labels is not None) and (args.dst_labels is not None):
        if len(args.src_labels) != len(args.dst_labels):
            print('len(args.src_labels) != len(args.dst_labels)')
            return
        label_map = {s: d for s, d in zip(args.src_labels, args.dst_labels)}
    is_json = False
    pred_anns = []
    id_pred_anns = {}
    anno = None
    ext = os.path.splitext(args.bbox_list_file)[-1].lower()
    if ext == '.json':
        is_json = True
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
                image_id = base_ids.get(image_id, image_id)
                if image_id is None:
                    continue
                info['image_id'] = image_id
                if label_map is not None:
                    label = label_map.get(info['category_id'], None)
                    if label is None:
                        continue
                    info['category_id'] = label
                else:
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
                        label = int(bbox[0])
                        if label_map is not None:
                            label = label_map.get(label, None)
                            if label is None:
                                continue
                        else:
                            label += label_offset
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

    if (not is_json) and (len(pred_anns) < 1):
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
                    label = int(bbox[0])
                    if label_map is not None:
                        label = label_map.get(label, None)
                        if label is None:
                            continue
                    else:
                        label += label_offset
                    bboxes.append([label, float(bbox[1]), float(bbox[2]), float(bbox[3]), float(bbox[4]), float(bbox[5])])
                if len(bboxes) > 0:
                    file_bboxes[file_name] = bboxes

    # try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    if True:
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
        scoreThrs = SidaCOCOeval.cal_range(args.score_thr_start, args.score_thr_end, args.score_thr_step)
        eval = SidaCOCOeval(anno, pred, 'bbox', scoreThrs=scoreThrs, iouThrs=args.iou_thrs, classwise=args.classwise)
        eval.evaluate()
        eval.accumulate()
        eval.summarize()
    # except Exception as e:
    #     print(f'pycocotools unable to run: {e}')

    print('Finish!')


if __name__ == '__main__':
    main()

