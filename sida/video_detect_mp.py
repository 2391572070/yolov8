#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@author:  yerunyuan
@contact: yerunyuan@163.com
"""

import numpy as np
import time

import cv2
import argparse
import os
import struct
import sys
import torch
import torch.nn.functional as F
from collections import OrderedDict
import multiprocessing
import random
import skvideo.io

sys.path.append('.')

seq_width = 28
seq_height = 28


def parse_args():
    parser = argparse.ArgumentParser(description='Video Detect')
    parser.add_argument('--base_path', type=str, required=True, help='video base path')
    parser.add_argument('--list_file', type=str, required=True, help='video list file')
    parser.add_argument('--out_path', type=str, required=True, help='out path')
    parser.add_argument('--checkpoint', type=str, required=True, help='checkpoint file path')
    parser.add_argument('--proccess_count', type=int, default=16, help='proccess count')
    parser.add_argument('--gpu_count', type=int, default=2, help='gpu count')
    parser.add_argument('--gpu_id_offset', type=int, default=0, help='gpu id offset')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='inference size h,w')
    parser.add_argument("--stride", type=int, default=32, help="image stride")
    parser.add_argument("--auto", action='store_true', help="auto")
    parser.add_argument('--conf_thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max_det', type=int, default=300, help='maximum detections per image')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic_nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--multi_label', action='store_true', help='multi label')
    parser.add_argument('--min_area', type=int, default=0, help='Bbox min area')
    parser.add_argument('--min_frame_count', type=int, default=0, help='statistic min frame count')
    parser.add_argument('--fps', type=int, default=None, help='fps')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--name', type=str, default='', help='out name')
    parser.add_argument("--feat_mode", type=int, default=0, help='save feat mode, 0 is all feat, 1 is only fuse label feat')
    parser.add_argument("--save_feat", action='store_true', help="save feat")
    parser.add_argument("--save_txt", action='store_true', help="save txt")
    parser.add_argument("--display", action='store_true', help="display detect results")
    parser.add_argument("--display_feat", action='store_true', help="display feat")
    parser.add_argument('--display_feat_labels', default=None, help='Space separated labels of the display feat ', nargs='*', type=int)
    parser.add_argument("--display_fuse_feat", action='store_true', help="display fuse feat")
    parser.add_argument('--display_fuse_mode', type=int, default=0, help='display fuse mode')
    parser.add_argument("--display_feat_use_softmax", action='store_true', help="display feat use softmax")
    parser.add_argument("--pause", action='store_true', help="pause display detect results")
    parser.add_argument('--seq_width', type=int, default=28, help='seq width')
    parser.add_argument('--seq_height', type=int, default=28, help='seq height')
    parser.add_argument("--statistic", action='store_true', help="statistic")
    parser.add_argument('--statistic_classes', nargs='+', type=int, default=None, help='statistic by class')
    parser.add_argument('--statistic_class_names', nargs='+', type=str, default=None, help='statistic class names')
    parser.add_argument('--statistic_confs', nargs='+', type=float, default=None, help='classes conf')
    parser.add_argument("--statistic_filter", action='store_true', help="statistic filter")
    parser.add_argument('--statistic_filter_min_obj_counts', nargs='+', type=int, default=None, help='statistic filter min obj counts')
    args = parser.parse_args()
    return args


args = parse_args()

random.seed(12345)


def os_system(cmd_str):
    if sys.platform.startswith('linux'):
        os.system(cmd_str)


def get_video_rotate(video_path):
    # skvideo.io.ffprobe可以读取视频的元信息，  返回一个 有序字典OrderedDict，
    # 如果 键“tag” 里面包含[OrderedDict([('@key', 'rotate'), ('@value', '90')])，
    # 则说明是需要做旋转的。 你可以根据需要在做判断是否旋转。
    metadata = skvideo.io.ffprobe(video_path)
    rotate = 0
    _video = metadata.get('video', None)
    if _video is not None:
        _tag = _video.get('tag', None)
        if _tag is not None:
            for t in _tag:
                _key = t.get('@key', None)
                _value = t.get('@value', None)
                if (_key is None) or (_value is None):
                    continue
                if _key == 'rotate':
                    rotate = int(_value)
                    break
    return rotate


def process_detect_func(results, frameids, score_thr, min_area, feat_mode):
    results, pred_feats, feat_shapes = results
    bbox_count = 0
    frameid_list = []
    label_list = []
    bboxes_list = []
    for result, frameid in zip(results, frameids):
        bboxes = result.cpu().numpy()

        if bboxes.size > 0:
            mask = bboxes[..., 4] >= score_thr
            bboxes = bboxes[mask]
            if bboxes.size > 0:
                scores = bboxes[..., 4]
                bboxes = bboxes.astype(dtype=np.int32)
                if min_area > 0:
                    area = np.prod(bboxes[..., 2:4] - bboxes[..., :2], 1)
                    vaild_mask = area > min_area
                    scores = scores[vaild_mask]
                    bboxes = bboxes[vaild_mask]
                    if bboxes.size < 1:
                        continue

                labels = bboxes[..., 5:6].tolist()
                bboxes = bboxes[..., :5].tolist()
                scores = scores.tolist()
                for i, score in enumerate(scores):
                    bboxes[i][4] = score

                _bbox_count = len(bboxes)
                frameid_list.extend([[frameid]] * _bbox_count)
                label_list.extend(labels)
                bboxes_list.extend(bboxes)
                bbox_count += _bbox_count

    results_dict = OrderedDict()
    results_dict['frame_id'] = frameid_list
    results_dict['label'] = label_list
    results_dict['bbox'] = bboxes_list
    results_dict['seg'] = None
    results_dict['frame_count'] = len(frameids)
    if pred_feats is not None:
        results_dict['feat_frameids'] = frameids
        if 0 < feat_mode:
            feat_sizes = [h*w for h, w in feat_shapes]
            pred_feats = pred_feats[:, 4:, ...]
            bs, n = pred_feats.shape[:2]
            pred_feats = pred_feats.split(feat_sizes, -1)
            feats = []
            align_h, align_w = feat_shapes[0]
            for i, (h, w) in enumerate(feat_shapes):
                feat = pred_feats[i].view(bs, n, h, w)
                if (h != align_h) or (w != align_w):
                    # feat = F.interpolate(feat, size=(align_h, align_w), mode='nearest')
                    feat = F.interpolate(feat, size=(align_h, align_w), mode='bilinear', align_corners=False)
                feats.append(feat)

            if 1 == feat_mode:
                feats = torch.stack(feats, -1)
                feats = torch.max(feats, -1)[0]
                feat_score, feat_label = torch.max(feats, 1)
            else:
                feat_count = len(feats)
                feat_score_labels = [torch.max(f, 1) for f in feats[::-1]]
                feat_score, feat_label = feat_score_labels[0]
                for i in range(1, feat_count):
                    _feat_score, _feat_label = feat_score_labels[i]
                    same_label_mask = feat_label == _feat_label
                    diff_label_mask = torch.logical_not(same_label_mask)
                    feat_score[same_label_mask] = torch.maximum(feat_score[same_label_mask],
                                                                _feat_score[same_label_mask])
                    if score_thr > 0:
                        score_mask = _feat_score >= score_thr
                        diff_label_mask.logical_and_(score_mask)
                    feat_score[diff_label_mask] = _feat_score[diff_label_mask]
                    feat_label[diff_label_mask] = _feat_label[diff_label_mask]

            feat_label = feat_label.to(dtype=feat_score.dtype)
            pred_feats = torch.stack((feat_label, feat_score), 1)
        results_dict['feat'] = pred_feats.cpu().share_memory_()

    return bbox_count, results_dict


def video_detect_run(file_queue, out_queue, pid, args):
    device = str((pid % args.gpu_count) + args.gpu_id_offset)

    import torch.backends.cudnn as cudnn

    from ultralytics.nn.autobackend import AutoBackend
    from ultralytics.yolo.utils import ops
    from ultralytics.yolo.utils.checks import check_imgsz
    from ultralytics.yolo.utils.torch_utils import select_device

    class Detection:
        def __init__(self, config, checkpoint=None, device='0'):
            imgsz = config.imgsz
            self.conf_thres = config.conf_thres
            self.iou_thres = config.iou_thres
            self.classes = config.classes
            self.agnostic_nms = config.agnostic_nms
            self.max_det = config.max_det
            self.multi_label = config.multi_label
            self.save_feat = config.save_feat
            self.device = select_device(device)
            with torch.no_grad():
                self.model = AutoBackend(checkpoint, device=self.device)
                self.model.eval()

            imgsz = check_imgsz(imgsz, stride=self.model.stride, min_dim=2)  # check image size
            with torch.no_grad():
                self.model(torch.zeros(1, 3, *imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

        def preprocess(self, img):
            img = img.to(self.device).float()
            # img = torch.from_numpy(img).to(self.device)
            # img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
            img /= 255  # 0 - 255 to 0.0 - 1.0
            return img

        def __call__(self, images, orig_w_hs):
            images = self.preprocess(images)
            preds = self.model(images)
            if isinstance(preds, (list, tuple)):
                preds = preds[0]
            feat_shapes = []
            ih, iw = images.shape[2:4]
            div = 8
            for _ in range(3):
                h, w = ih // div, iw // div
                feat_shapes.append((h, w))
                div *= 2

            if self.save_feat:
                pred_feats = preds.half()
            preds = ops.non_max_suppression(preds, self.conf_thres, self.iou_thres, agnostic=self.agnostic_nms,
                                            multi_label=self.multi_label, max_det=self.max_det, classes=self.classes)
            shape = images.shape[2:]
            results = []
            for pred, orig_w_h in zip(preds, orig_w_hs):
                orig_shape = [orig_w_h[1], orig_w_h[0]]
                pred[:, :4] = ops.scale_boxes(shape, pred[:, :4], orig_shape).round()
                results.append(pred)
            if self.save_feat:
                return results, pred_feats, feat_shapes
            else:
                return results, None, None

    def init_model_func(args):
        detecter = Detection(args, checkpoint=args.checkpoint, device=device)
        return detecter

    try:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        # Initialize model
        model = init_model_func(args)

        with torch.no_grad():
            while True:
                file_bboxes = file_queue.get(block=True)
                if file_bboxes is None:
                    break

                (file_name, index, images, frameids, w_hs), _, _ = file_bboxes

                if images is not None:
                    # images.requires_grad = False
                    # images = images.cuda(non_blocking=True)
                    results = model(images, w_hs)
                    results = process_detect_func(results, frameids, args.conf_thres, args.min_area, args.feat_mode)
                else:
                    results = None

                while out_queue.qsize() > 100:
                    time.sleep(0.01)
                out_queue.put(((file_name, results, index), False, pid))
    except Exception as e:
        if str(e) != '':
            print('video_detect_run', e)

    out_queue.put((None, True, pid))


def video_image_run(file_queue, out_queue, pid, args):
    try:

        from ultralytics.yolo.data.augment import LetterBox

        class DetectTransform:

            def __init__(self, config):
                # prepare data
                imgsz = config.imgsz
                if isinstance(imgsz, (tuple, list)):
                    if len(imgsz) > 1:
                        self.imgsz = imgsz
                    else:
                        self.imgsz = [imgsz[0], imgsz[0]]
                else:
                    self.imgsz = [imgsz, imgsz]
                self.img_transform = None

                self.letter_box = LetterBox(self.imgsz, config.auto, stride=config.stride)

            def __call__(self, image):
                pad_image = self.letter_box(image=image)
                # cv2.imshow('image', image)
                # cv2.imshow('pad_image', pad_image)
                # cv2.waitKey()

                # Convert
                image = pad_image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                image = np.ascontiguousarray(image)
                image = torch.from_numpy(image)

                return image

        def put_images(out_queue, file_name, index, images, frameids, w_hs):
            while out_queue.qsize() > 64:
                time.sleep(0.01)
            out_queue.put(((file_name, index, images, frameids, w_hs), False, pid))

        def pre_batch_image(out_queue, file_name, batch_index, batch_images, batch_frameids, batch_w_hs):
            image_count = len(batch_images)
            if image_count > 0:
                input_images = torch.stack(batch_images, 0)
                input_images.share_memory_()
                input_frameids = batch_frameids.copy()
                input_w_hs = batch_w_hs.copy()
                put_images(out_queue, file_name, batch_index, input_images, input_frameids, input_w_hs)
                batch_index += 1
                batch_images.clear()
                batch_frameids.clear()
                batch_w_hs.clear()
            return batch_index

        min_frame_count = args.min_frame_count
        batch_size = args.batch_size
        detect_trans = DetectTransform(args)
        while True:
            file_bboxes = file_queue.get(block=True)
            if file_bboxes is None:
                break

            file_name, video_path = file_bboxes

            cap = cv2.VideoCapture(video_path)
            total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frame_count < min_frame_count:
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            if args.fps is None:
                interval_frame = 1
            else:
                if fps <= args.fps:
                    interval_frame = 1
                else:
                    interval_frame = fps / args.fps

            rotate = get_video_rotate(video_path)
            frame_count = 0
            write_frame_count = 0

            batch_frameids = []
            batch_images = []
            batch_w_hs = []
            batch_index = 0
            while True:
                grabbed, image_bgr = cap.read()

                if not grabbed:
                    batch_index = pre_batch_image(out_queue, file_name, batch_index, batch_images, batch_frameids, batch_w_hs)
                    break

                frame_count += 1
                if frame_count > write_frame_count:
                    write_frame_count += interval_frame
                else:
                    continue

                if rotate == 90:
                    image_bgr = cv2.flip(cv2.transpose(image_bgr), 1)
                elif rotate == 180:
                    image_bgr = cv2.flip(image_bgr, -1)
                elif rotate == 270:
                    image_bgr = cv2.flip(cv2.transpose(image_bgr), 0)

                # image_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
                image = detect_trans(image_bgr)
                batch_images.append(image)
                batch_frameids.append(frame_count)
                batch_w_hs.append([image_bgr.shape[1], image_bgr.shape[0]])
                batch_image_count = len(batch_images)
                is_end = frame_count >= total_frame_count
                if (batch_image_count >= batch_size) or is_end:
                    batch_index = pre_batch_image(out_queue, file_name, batch_index, batch_images, batch_frameids, batch_w_hs)
                    if is_end:
                        break

            if batch_index > 0:
                put_images(out_queue, file_name, batch_index, None, None, None)
    except Exception as e:
        if str(e) != '':
            print('video_image_run', e)


def out_run(out_queue, args, out_file_path, total_file_count):
    save_txt = args.save_txt
    out_dir = os.path.dirname(out_file_path)
    out_file = open(out_file_path, 'a+', encoding='utf-8')
    file_count = 0
    batch_count = 0
    try:
        has_write_title = False
        finish_worker_count = 0
        file_results = {}
        file_result_count = {}
        begin_time = start_time = time.time()
        frame_count = 0
        total_frame_count = 0
        while True:
            file_info = out_queue.get(block=True)
            if file_info is None:
                break
            results, finish, pid = file_info
            if finish:
                print('Proc{} finish'.format(pid, ))
                finish_worker_count += 1
                if args.gpu_count <= finish_worker_count:
                    break
                continue

            if results is not None:
                file_name, _results, index = results
                if _results is None:  # 最后一个batch，记录索引
                    file_result_count[file_name] = index
                else:
                    obj_count, results_dict = _results
                    batch_count += 1
                    frame_count += results_dict.pop('frame_count', 0)
                    if frame_count > 100:
                        total_frame_count += frame_count
                        end_time = time.time()
                        fps1 = total_frame_count / (end_time - begin_time)
                        fps2 = frame_count / (end_time - start_time)
                        start_time = end_time
                        frame_count = 0
                        if file_count > 0:
                            _total_frame_count = (total_frame_count//file_count)*total_file_count
                        else:
                            _total_frame_count = total_frame_count
                        print('{} {}/{} {} {} Process Id {}: {} - {}, FPS {:.3f} {:.3f}'.format((end_time - begin_time), file_count, total_file_count, total_frame_count, _total_frame_count, pid, file_name, index, fps1, fps2))

                    feat_frameids = results_dict.pop('feat_frameids', None)
                    feats = results_dict.pop('feat', None)

                    segs = results_dict.pop('seg', None)
                    results_info = file_results.get(file_name, None)
                    if results_info is None:
                        out_base_name = os.path.join(out_dir, os.path.splitext(file_name)[0])
                        dirname = os.path.dirname(out_base_name)
                        if not os.path.exists(dirname):
                            os.makedirs(dirname)
                        if save_txt:
                            out_bbox_file_path = out_base_name + '.box.txt'
                            out_bbox_file = open(out_bbox_file_path, 'w')
                        else:
                            out_bbox_file_path = out_base_name + '.box.dat'
                            out_bbox_file = open(out_bbox_file_path, 'wb')
                        if segs is not None:
                            out_seq_file_path = out_base_name + '.seq.dat'
                            out_seq_file = open(out_seq_file_path, 'wb')
                        else:
                            out_seq_file = None
                        if feats is not None:
                            out_feat_file_path = out_base_name + '.feat.npz'
                            out_feat_infos = [out_feat_file_path, []]
                        else:
                            out_feat_infos = None
                        file_results[file_name] = [1, obj_count, out_bbox_file, out_seq_file, out_feat_infos]
                        if obj_count > 0:
                            results_name = results_dict.keys()
                            if not has_write_title:
                                has_write_title = True
                                out_title_file_path = os.path.splitext(out_file_path)[0] + '.title.txt'
                                with open(out_title_file_path, 'w', encoding='utf-8') as out_title_file:
                                    line_str = (';'.join(results_name)) + '\n'
                                    out_title_file.write(line_str)
                                    results_count = []
                                    for name in results_name:
                                        results_count.append(len(results_dict[name][0]))
                                    line_str = (';'.join(map(str, results_count))) + '\n'
                                    out_title_file.write(line_str)
                                os_system('chmod a+wr \"{}\"'.format(out_title_file_path))
                    else:
                        results_info[0] += 1
                        results_info[1] += obj_count
                        out_bbox_file = results_info[2]
                        out_seq_file = results_info[3]
                        out_feat_infos = results_info[4]
                    # write
                    if out_bbox_file:
                        for i in range(obj_count):
                            results_info = []
                            for name, _results in results_dict.items():
                                results_info.extend(_results[i])
                            if save_txt:
                                dat = '{},{},{},{},{},{},{:.04f}\n'.format(*results_info)
                            else:
                                dat = struct.pack('6i1f', *results_info)
                            # print(len(dat), dat_list)
                            out_bbox_file.write(dat)
                    if (out_seq_file is not None) and (segs is not None):
                        segs = segs.numpy()
                        # print(segs.dtype, segs.shape)
                        segs.tofile(out_seq_file)
                    if (out_feat_infos is not None) and (feats is not None) and (feat_frameids is not None):
                        out_feat_infos[1].append([feat_frameids, feats.numpy()])

                for file_name, result_count in file_result_count.items():
                    results_info = file_results.get(file_name, None)
                    if results_info is None:
                        continue

                    if results_info[0] >= result_count:
                        out_bbox_file = results_info[2]
                        if out_bbox_file is not None:
                            out_bbox_file.close()
                        out_seq_file = results_info[3]
                        if out_seq_file is not None:
                            out_seq_file.close()
                        out_feat_infos = results_info[4]
                        if out_feat_infos is not None:
                            if len(out_feat_infos) > 0:
                                out_feat_file_path = out_feat_infos[0]
                                out_feat_info_list = out_feat_infos[1]
                                out_feat_info_list.sort(key=lambda x: x[0][0])
                                out_frameids = []
                                out_feats = []
                                for frameids, feats in out_feat_info_list:
                                    out_frameids.extend(frameids)
                                    out_feats.extend(feats)
                                out_feats = np.stack(out_feats, 0)
                                np.savez_compressed(out_feat_file_path, frameids=out_frameids, feats=out_feats)
                                out_frameids = None
                                out_feats = None
                        file_count += 1
                        line_str = '{},{}\n'.format(file_name, results_info[1])
                        out_file.write(line_str)
                        out_file.flush()
                        file_results.pop(file_name)
    except Exception as e:
        print('out_run', e)
    out_file.close()
    os_system('chmod a+wr \"{}\" -Rf'.format(out_dir))


def video_detect_mp(args):
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
        os_system('chmod a+wr \"{}\"'.format(args.out_path))
    manager = multiprocessing.Manager()
    file_queue = manager.Queue()
    out_queue = manager.Queue()
    image_queue = manager.Queue()

    out_file_path = os.path.join(args.out_path, 'video_detect_bbox.txt')
    filter_file_names = set()
    if os.path.exists(out_file_path):
        try:
            bbox_count = 0
            with open(out_file_path, 'r', encoding='utf-8') as file:
                for line in file.readlines():
                    lines = line.strip().split(',')
                    file_name = lines[0]
                    filter_file_names.add(file_name)
                    bbox_count += int(lines[1])
        except Exception as e:
            print(e)
            print('Filter File Error!')
            return

    file_list = []
    with open(args.list_file, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            file_name = line.strip().split()[0]
            if file_name in filter_file_names:
                continue
            file_list.append(file_name)
    total_file_count = len(file_list)
    print('Total File Count:', total_file_count)
    if total_file_count < 1:
        return

    start_time = time.time()

    workers = []
    for i in range(args.proccess_count):
        workers.append(multiprocessing.Process(target=video_image_run, args=(file_queue, image_queue, i, args)))

    det_workers = []
    for i in range(args.gpu_count):
        det_workers.append(multiprocessing.Process(target=video_detect_run, args=(image_queue, out_queue, i, args)))

    out_worker = multiprocessing.Process(target=out_run, args=(out_queue, args, out_file_path, total_file_count))
    out_worker.start()

    for worker in workers:
        worker.start()

    for det_worker in det_workers:
        det_worker.start()

    file_count = 0
    video_dir = args.base_path
    for file_name in file_list:
        while file_queue.qsize() >= args.proccess_count*4:
            time.sleep(0.01)
        if file_name in filter_file_names:
            continue
        video_path = os.path.join(video_dir, file_name)
        if not os.path.exists(video_path):
            continue
        # 预读取
        with open(video_path, 'rb') as _file:
            _file_buffer = _file.read()
        file_queue.put((file_name, video_path))
        file_count += 1
        # if file_count >= 16:
        #     break

    for i in range(args.proccess_count):
        file_queue.put(None)

    for worker in workers:
        worker.join()

    for i in range(args.gpu_count):
        image_queue.put(None)

    for det_worker in det_workers:
        det_worker.join()

    out_worker.join()

    end_time = time.time()
    print('use time: {:.03f}s'.format(end_time-start_time))

    # video_detect_run(file_queue, out_queue, 0, args, file_lists[0])
    print('finish!')


def random_color(color_list=[], min_color_value=50):
    while True:
        # color = (25*random.randint(1, 10), 25*random.randint(1, 10), 25*random.randint(1, 10))
        color = (25 * random.randint(0, 10), 25 * random.randint(0, 10), 25 * random.randint(0, 10))
        # color = (50 * random.randint(0, 5), 50 * random.randint(0, 5), 50 * random.randint(0, 5))
        if max(color) < min_color_value:
            continue
        if color in color_list:
            continue
        else:
            break
    # print(color)
    return color


class _CalPadding:
    """Resize image and padding for detection, instance segmentation, pose"""

    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, stride=32):
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride

    def __call__(self, image_shape):
        shape = image_shape  # current shape [height, width]
        new_shape = self.new_shape
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        padding_info = (left, top, right, bottom), (new_unpad[0], new_unpad[1]), (
        new_unpad[0] + left + right, new_unpad[1] + top + bottom)
        # print('CalPadding:', padding_info)
        return padding_info


class Padding:

    def __init__(self, config):
        # prepare data
        imgsz = config.imgsz
        if isinstance(imgsz, (tuple, list)):
            if len(imgsz) > 1:
                self.imgsz = imgsz
            else:
                self.imgsz = [imgsz[0], imgsz[0]]
        else:
            self.imgsz = [imgsz, imgsz]
        self.img_transform = None

        self._cal_padding = _CalPadding(self.imgsz, config.auto, stride=config.stride)
        from ultralytics.yolo.data.augment import LetterBox
        self._letter_box = LetterBox(self.imgsz, config.auto, stride=config.stride)

    def cal_padding(self, width, height):
        return self._cal_padding((height, width))

    def __call__(self, image):
        return self._letter_box(image=image)


def video_detect_display(args):
    from ultralytics.yolo.utils.plotting import Colors

    opacity = 0.5

    out_file_path = os.path.join(args.out_path, 'video_detect_bbox.txt')
    file_name_bbox_counts = []
    if os.path.exists(out_file_path):
        try:
            with open(out_file_path, 'r', encoding='utf-8') as file:
                for line in file.readlines():
                    lines = line.strip().split(',')
                    file_name = lines[0]
                    file_name_bbox_counts.append([file_name, int(lines[1])])
        except Exception as e:
            print(e)
            print('Filter File Error!')
            return

    seq_width = args.seq_width
    seq_height = args.seq_height
    seq_size = seq_width*seq_height

    multi_label = args.multi_label
    display_feat = args.display_feat
    display_feat_use_softmax = args.display_feat_use_softmax
    display_fuse_feat = args.display_fuse_feat
    feat_mode = args.feat_mode
    display_fuse_mode = args.display_fuse_mode

    score_thr = args.conf_thres
    pause = args.pause
    save_txt = args.save_txt
    display_feat_labels = args.display_feat_labels
    if display_feat_labels is not None:
        display_feat_labels = set(display_feat_labels)
        if len(display_feat_labels) < 1:
            display_feat_labels = None

    padding = Padding(args)

    # color_list = []
    color_map = {}

    colors = Colors()
    color_tensor = None

    main_win_name = 'video detect'
    cv2.namedWindow(main_win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    video_dir = args.base_path
    out_dir = os.path.dirname(out_file_path)
    for file_name, bbox_count in file_name_bbox_counts:
        video_path = os.path.join(video_dir, file_name)
        if not os.path.exists(video_path):
            continue

        file_base_name = os.path.splitext(file_name)[0]
        if save_txt:
            bbox_file_path = os.path.join(out_dir, file_base_name + '.box.txt')
            if not os.path.exists(bbox_file_path):
                continue
            bbox_file = open(bbox_file_path, 'r')
        else:
            bbox_file_path = os.path.join(out_dir, file_base_name + '.box.dat')
            if not os.path.exists(bbox_file_path):
                continue
            bbox_file = open(bbox_file_path, 'rb')
        seq_file_path = os.path.join(out_dir, file_base_name + '.seq.dat')
        if os.path.exists(seq_file_path):
            seq_file = open(seq_file_path, 'rb')
        else:
            seq_file = None

        if display_feat:
            feat_file_path = os.path.join(out_dir, file_base_name + '.feat.npz')
            if os.path.exists(feat_file_path):
                feat_infos = np.load(feat_file_path)
                frameids = feat_infos['frameids']
                feats = feat_infos['feats']
                feat_infos = {frameid: feat for frameid, feat in zip(frameids, feats)}
            else:
                feat_infos = None

        display_infos = {}
        for _ in range(bbox_count):
            if save_txt:
                bbox_info = bbox_file.readline().strip()
                if len(bbox_info) < 12:
                    break
                bbox_info = bbox_info.split(',')
                bbox_info = [int(bbox_info[0]), int(bbox_info[1]), int(bbox_info[2]), int(bbox_info[3]), int(bbox_info[4]), int(bbox_info[5]), float(bbox_info[6])]
            else:
                bbox_info = bbox_file.read(28)
                if len(bbox_info) < 28:
                    break
                bbox_info = struct.unpack('6i1f', bbox_info)
            frame_id = bbox_info[0]
            label = bbox_info[1]
            score = bbox_info[6]
            if seq_file is not None:
                seq_info = seq_file.read(seq_size)
            else:
                seq_info = None
            if score < score_thr:
                continue

            if seq_info is not None:
                seq = np.frombuffer(seq_info, dtype=np.bool, count=seq_size).reshape(seq_height, seq_width)
            else:
                seq = None
            display_info = display_infos.get(frame_id, None)
            if display_info is None:
                display_infos[frame_id] = [[label, bbox_info[2:6], score, seq]]
            else:
                display_info.append([label, bbox_info[2:6], score, seq])

        cap = cv2.VideoCapture(video_path)
        total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        rotate = get_video_rotate(video_path)
        if (rotate == 90) or (rotate == 270):
            width, height = height, width

        padding_info = padding.cal_padding(width, height)
        left, top, right, bottom = padding_info[0]
        ow, oh = padding_info[1]
        pw, ph = padding_info[2]
        layer_sizes = []
        layer_shapes = []
        align_w = None
        align_h = None
        for s in [8, 16, 32]:
            h, w = ph // s, pw // s
            layer_sizes.append(h*w)
            layer_shapes.append((h, w))
            if align_h is None:
                align_h = h
            else:
                if align_h < h:
                    align_h = h
            if align_w is None:
                align_w = w
            else:
                if align_w < w:
                    align_w = w
        frame_count = 0
        while True:
            grabbed, image_bgr = cap.read()
            if not grabbed:
                break

            frame_count += 1
            display_info = display_infos.get(frame_count, None)
            # if display_info is None:
            #     continue

            if rotate == 90:
                image_bgr = cv2.flip(cv2.transpose(image_bgr), 1)
            elif rotate == 180:
                image_bgr = cv2.flip(image_bgr, -1)
            elif rotate == 270:
                image_bgr = cv2.flip(cv2.transpose(image_bgr), 0)

            # height, width = image_bgr.shape[0:2]
            if display_info is not None:
                for label, bbox, score, seq in display_info:
                    color = color_map.get(label, None)
                    if color is None:
                        # color = random_color(color_list)
                        color = colors(label)
                        color_map[label] = color
                    if seq is not None:
                        seq = cv2.resize(seq.astype(dtype=np.uint8), (int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])), interpolation=cv2.INTER_NEAREST).astype(dtype=np.bool)
                        roi_image = image_bgr[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                        roi_image[seq] = (roi_image[seq]*(1-opacity) + opacity*np.array(color, dtype=np.uint8)).astype(np.uint8)
                    cv2.rectangle(image_bgr, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness=2)
                    cv2.putText(image_bgr, '{} {:.2f}'.format(label, score), (bbox[0], bbox[1]-4), cv2.FONT_HERSHEY_COMPLEX, 1, color, thickness=2)

            cv2.putText(image_bgr, '{}/{}'.format(frame_count, total_frame_count), (8, 64), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), thickness=2)

            if display_feat:
                image_bgr = padding(image_bgr)
                if feat_infos is not None:
                    feat = feat_infos.get(frame_count, None)
                    if feat is not None:
                        if 0 == feat_mode:
                            if color_tensor is None:
                                color_tensor = torch.tensor([colors(i) for i in range(feat.shape[0])], dtype=torch.uint8)
                                _colors = color_tensor[:, None].numpy()
                            feat = torch.from_numpy(feat[4:, ...]).float()
                            if multi_label:
                                feats = []
                                _feats = feat.split(layer_sizes, -1)
                                for i, layer_shape in enumerate(layer_shapes):
                                    _feat = _feats[i].view(-1, *layer_shape)
                                    if (layer_shape[0] != align_h) or (layer_shape[1] != align_w):
                                        # _feat = F.interpolate(_feat[None], size=(align_h, align_w), mode='nearest')[0]
                                        _feat = F.interpolate(_feat[None], size=(align_h, align_w), mode='bilinear', align_corners=False)[0]
                                    feats.append(_feat)
                                feats = torch.stack(feats, -1)
                                feats = torch.max(feats, -1)[0]
                                factors = F.interpolate((1-feats)[None], size=(image_bgr.shape[0], image_bgr.shape[1]), mode='bilinear', align_corners=False)[0][:,:,:,None].numpy()
                                feats = feats.numpy()

                                feat_images = []
                                fuse_feat_images = []
                                for i, _feat in enumerate(feats):
                                    if display_feat_labels is not None:
                                        if i not in display_feat_labels:
                                            continue
                                    color = _colors[i]
                                    feat_image = (_feat[:, :, None] * color).astype(np.uint8)
                                    feat_images.append(feat_image)
                                    # feat_image = cv2.resize(feat_image, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
                                    feat_image = cv2.resize(feat_image, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
                                    fuse_feat_image = (factors[i]*image_bgr + feat_image).astype(np.uint8)
                                    fuse_feat_images.append(fuse_feat_image)
                                    cv2.imshow('feat{}'.format(i), feat_image)
                                    cv2.imshow('fuse{}'.format(i), fuse_feat_image)
                            elif display_fuse_feat:
                                feats = []
                                _feats = feat.split(layer_sizes, -1)
                                for i, layer_shape in enumerate(layer_shapes):
                                    _feat = _feats[i].view(-1, *layer_shape)
                                    if (layer_shape[0] != align_h) or (layer_shape[1] != align_w):
                                        # _feat = F.interpolate(_feat[None], size=(align_h, align_w), mode='nearest')[0]
                                        _feat = F.interpolate(_feat[None], size=(align_h, align_w), mode='bilinear', align_corners=False)[0]
                                    feats.append(_feat)

                                if display_fuse_mode == 0:
                                    feat_count = len(feats)
                                    feat_score_labels = [torch.max(f, 0) for f in feats[::-1]]
                                    feat_score, feat_label = feat_score_labels[0]
                                    for i in range(1, feat_count):
                                        _feat_score, _feat_label = feat_score_labels[i]
                                        same_label_mask = feat_label == _feat_label
                                        diff_label_mask = torch.logical_not(same_label_mask)
                                        feat_score[same_label_mask] = torch.maximum(feat_score[same_label_mask], _feat_score[same_label_mask])
                                        if score_thr > 0:
                                            score_mask = _feat_score >= score_thr
                                            diff_label_mask.logical_and_(score_mask)
                                        feat_score[diff_label_mask] = _feat_score[diff_label_mask]
                                        feat_label[diff_label_mask] = _feat_label[diff_label_mask]
                                else:
                                    feats = torch.stack(feats, -1)
                                    feats = torch.max(feats, -1)[0]
                                    feat_score, feat_label = torch.max(feats, 0)

                                feat_image = (color_tensor[feat_label] * feat_score[:, :, None])
                                if score_thr > 0:
                                    feat_image[feat_score < score_thr] = 0
                                feat_image = feat_image.numpy().astype(np.uint8)
                                feat_image = cv2.resize(feat_image, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
                                alpha = F.interpolate((1 - feat_score)[None, None], size=(image_bgr.shape[0], image_bgr.shape[1]), mode='bilinear', align_corners=False)[0][0][:, :, None].numpy()
                                fuse_feat_image = (alpha * image_bgr + feat_image).astype(np.uint8)
                                cv2.imshow('feat', feat_image)
                                cv2.imshow('fuse', fuse_feat_image)
                            else:
                                if display_feat_use_softmax:
                                    feat = torch.softmax(feat, 0)
                                feat_score, feat_label = torch.max(feat, 0)
                                # feat_label = torch.argmax(feat, 0)
                                feat_images = color_tensor[feat_label]
                                if score_thr > 0:
                                    feat_images[feat_score < score_thr] = 0
                                # feat_images[feat_label!=6] = 0
                                feat_images = feat_images.split(layer_sizes, 0)
                                _feat_images = []
                                for i, layer_shape in enumerate(layer_shapes):
                                    feat_image = feat_images[i].view(*layer_shape, -1).numpy()
                                    if (layer_shape[0] != align_h) or (layer_shape[1] != align_w):
                                        feat_image = cv2.resize(feat_image, (align_w, align_h), interpolation=cv2.INTER_NEAREST)
                                        # feat_image = cv2.resize(feat_image, (align_w, align_h), interpolation=cv2.INTER_LINEAR)
                                        # feat_image = F.interpolate(feat_image, (align_h, align_w, feat_image.shape[-1]), mode='nearest')
                                        # feat_image = F.interpolate(feat_image, (align_h, align_w, feat_image.shape[-1]), mode='bilinear', align_corners=False)
                                    _feat_images.append(feat_image)
                                    cv2.imshow('feat{}'.format(i), feat_image)

                                feat_images = _feat_images[::-1]
                                fuse_feat_image = None
                                for feat_image in feat_images:
                                    if fuse_feat_image is None:
                                        fuse_feat_image = feat_image
                                    else:
                                        mask = feat_image.sum(axis=-1) > 0
                                        fuse_feat_image[mask] = feat_image[mask]
                                fuse_feat_image = cv2.resize(fuse_feat_image, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
                                cv2.imshow('fuse feat', fuse_feat_image)
                                mask = 0 != fuse_feat_image
                                image_bgr[mask] = ((1-opacity)*image_bgr[mask] + opacity*fuse_feat_image[mask]).astype(np.uint8)
                        elif 0 < feat_mode:
                            if color_tensor is None:
                                color_tensor = torch.tensor([colors(i) for i in range(128)], dtype=torch.uint8)
                            feat = torch.from_numpy(feat)
                            feat_label = feat[0].long()
                            feat_score = feat[1].float()
                            feat_image = color_tensor[feat_label] * feat_score[:, :, None]
                            if score_thr > 0:
                                feat_image[feat_score < score_thr] = 0
                            feat_image = feat_image.numpy().astype(np.uint8)
                            feat_image = cv2.resize(feat_image, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
                            alpha = F.interpolate((1 - feat_score)[None, None], size=(image_bgr.shape[0], image_bgr.shape[1]), mode='bilinear', align_corners=False)[0][0][:, :, None].numpy()
                            fuse_feat_image = (alpha * image_bgr + feat_image).astype(np.uint8)
                            cv2.imshow('feat', feat_image)
                            cv2.imshow('fuse', fuse_feat_image)
            else:
                image_bgr = cv2.resize(image_bgr, (width // 2, height // 2))
            print(frame_count)
            cv2.imshow(main_win_name, image_bgr)
            if pause:
                key = cv2.waitKey()
            else:
                key = cv2.waitKey(1)
            if key == 27:
                return


def video_detect_statistic(args):
    min_frame_count = args.min_frame_count
    statistic_class_names = args.statistic_class_names
    statistic_classes = args.statistic_classes
    if statistic_classes is not None:
        if len(statistic_classes) < 1:
            statistic_classes = None
            statistic_class_names = None
        else:
            if statistic_class_names is not None:
                if len(statistic_class_names) != len(statistic_classes):
                    statistic_class_names = None
                else:
                    statistic_class_names = {c: n for c, n in zip(statistic_classes, statistic_class_names)}
            # class_to_indexes = {c: i for i, c in enumerate(statistic_classes)}
            index_to_classes = {i: c for i, c in enumerate(statistic_classes)}
            statistic_classes = set(statistic_classes)

    score_thr = args.conf_thres
    if statistic_classes is None:
        class_score_thrs = {}
        statistic_class_names = None
    else:
        statistic_confs = args.statistic_confs
        if statistic_confs is not None:
            count0 = len(statistic_confs)
            if count0 < 1:
                class_score_thrs = {}
            else:
                count1 = len(statistic_classes)
                count = count1 - count0
                if count > 0:
                    for _ in range(count):
                        statistic_confs.append(score_thr)
                class_score_thrs = {c: statistic_confs[i] for i, c in index_to_classes.items()}
        else:
            class_score_thrs = {}

    out_file_path = os.path.join(args.out_path, 'video_detect_bbox.txt')
    file_name_bbox_counts = []
    if os.path.exists(out_file_path):
        try:
            with open(out_file_path, 'r', encoding='utf-8') as file:
                for line in file.readlines():
                    lines = line.strip().split(',')
                    file_name = lines[0]
                    file_name_bbox_counts.append([file_name, int(lines[1])])
        except Exception as e:
            print(e)
            print('Filter File Error!')
            return

    save_txt = args.save_txt
    file_count = 0
    file_class_count_infos = {}
    out_dir = os.path.dirname(out_file_path)
    for file_name, bbox_count in file_name_bbox_counts:
        file_count += 1
        if (file_count % 100) == 0:
            print('Statistic File Count:', file_count)
            # break

        if save_txt:
            bbox_file_path = os.path.join(out_dir, os.path.splitext(file_name)[0] + '.box.txt')
            if not os.path.exists(bbox_file_path):
                continue
            bbox_file = open(bbox_file_path, 'r')
        else:
            bbox_file_path = os.path.join(out_dir, os.path.splitext(file_name)[0] + '.box.dat')
            if not os.path.exists(bbox_file_path):
                continue
            bbox_file = open(bbox_file_path, 'rb')

        class_infos = {}
        for _ in range(bbox_count):
            if save_txt:
                bbox_info = bbox_file.readline().strip()
                if len(bbox_info) < 12:
                    break
                bbox_info = bbox_info.split(',')
                bbox_info = [int(bbox_info[0]), int(bbox_info[1]), int(bbox_info[2]), int(bbox_info[3]), int(bbox_info[4]), int(bbox_info[5]), float(bbox_info[6])]
            else:
                bbox_info = bbox_file.read(28)
                if len(bbox_info) < 28:
                    break
                bbox_info = struct.unpack('6i1f', bbox_info)
            frame_id = bbox_info[0]
            label = bbox_info[1]
            if statistic_classes is not None:
                if label not in statistic_classes:
                    continue
            score = bbox_info[6]
            if score < class_score_thrs.get(label, score_thr):
                continue
            class_info = class_infos.get(label, None)
            if class_info is None:
                class_infos[label] = {frame_id: 1}
            else:
                max_object_count = class_info.get(frame_id, 0)
                max_object_count += 1
                class_info[frame_id] = max_object_count

        class_count_infos = {}
        for c, class_info in class_infos.items():
            frame_count = len(class_info)
            if frame_count < min_frame_count:
                continue
            max_object_count = max(class_info.values())
            class_count_infos[c] = [max_object_count, frame_count]

        if len(class_count_infos) > 0:
            file_class_count_infos[file_name] = class_count_infos

    if (file_count % 100) != 0:
        print('Statistic BBox File Count:', file_count)

    file_names = list(file_class_count_infos.keys())
    file_names.sort()
    file_count = 0
    class_file_counts = {}
    out_file_path = os.path.join(args.out_path, 'class_count_statistic_infos.txt')
    with open(out_file_path, 'w', encoding='utf-8') as file:
        for file_name in file_names:
            file_count += 1
            if (file_count % 100) == 0:
                print('Output File Count:', file_count)
            class_count_infos = file_class_count_infos[file_name]
            classes = list(class_count_infos.keys())
            classes.sort()
            out_str_list = []
            for c in classes:
                max_object_count, frame_count = class_count_infos[c]
                out_str_list.append('{},{},{}'.format(c, max_object_count, frame_count))
                class_file_counts[c] = class_file_counts.get(c, 0) + 1
            out_str = ';'.join(out_str_list)
            file.write('{} {}\n'.format(file_name, out_str))
    if (file_count % 100) != 0:
        print('Output File Count:', file_count)
    os_system('chmod a+wr \"{}\"'.format(out_file_path))

    classes = list(class_file_counts.keys())
    classes.sort()
    print('=' * 80)
    for c in classes:
        if statistic_class_names is None:
            print(c, class_file_counts[c])
        else:
            print(statistic_class_names.get(c, 'Unknow'), c, class_file_counts[c])
    print('=' * 80)
    print('Finish!')


def video_detect_statistic_filter(args):
    min_frame_count = args.min_frame_count
    min_obj_counts = args.statistic_filter_min_obj_counts
    statistic_class_names = args.statistic_class_names
    statistic_classes = args.statistic_classes
    if statistic_classes is not None:
        if len(statistic_classes) < 1:
            statistic_classes = None
        else:
            if statistic_class_names is not None:
                if len(statistic_class_names) != len(statistic_classes):
                    statistic_class_names = None
                else:
                    statistic_class_names = {c: n for c, n in zip(statistic_classes, statistic_class_names)}

            if min_obj_counts is not None:
                if len(min_obj_counts) != len(statistic_classes):
                    min_obj_counts = None
                else:
                    min_obj_counts = {c: n for c, n in zip(statistic_classes, min_obj_counts)}

            statistic_classes = set(statistic_classes)

    if statistic_classes is None:
        min_obj_counts = None
        statistic_class_names = None

    if (min_obj_counts is None) or isinstance(min_obj_counts, list):
        min_obj_counts = {}

    out_lines = []
    file_count = 0
    file_path = os.path.join(args.out_path, 'class_count_statistic_infos.txt')
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            file_count += 1
            if (file_count % 100) == 0:
                print('File Count:', file_count)
            # 290b100001051e000017373443595350/video1/20220524/20220524_140601.avi 0,2,838;1,2,530;11,1,136
            lines = line.strip().split()
            if len(lines) < 2:
                continue
            # file_name = lines[0]
            info = lines[1]
            infos = info.split(';')

            need_keep = False
            for info in infos:
                label, max_object_count, frame_count = list(map(int, info.split(',')))
                if statistic_classes is not None:
                    if label not in statistic_classes:
                        continue
                if max_object_count < min_obj_counts.get(label, 1):
                    continue
                if frame_count < min_frame_count:
                    continue
                need_keep = True
                break
            if need_keep:
                out_lines.append(line)
    if (file_count % 100) != 0:
        print('File Count:', file_count)

    if statistic_classes is None:
        sub_file_name = 'all'
    else:
        statistic_classes = list(statistic_classes)
        statistic_classes.sort()
        if statistic_class_names is None:
            sub_file_name = '_'.join(map(str, statistic_classes))
        else:
            sub_file_name = '_'.join([statistic_class_names[c] for c in statistic_classes])

    file_count = 0
    out_file_path = os.path.join(args.out_path, 'class_count_statistic_infos_{}.txt'.format(sub_file_name))
    with open(out_file_path, 'w', encoding='utf-8') as file:
        for line in out_lines:
            file_count += 1
            if (file_count % 100) == 0:
                print('Output File Count:', file_count)
            file.write(line)
    if (file_count % 100) != 0:
        print('Output File Count:', file_count)
    os_system('chmod a+wr \"{}\"'.format(out_file_path))

    print(sub_file_name, len(out_lines))
    print('Finish!')


if __name__ == "__main__":
    if args.display:
        video_detect_display(args)
    elif args.statistic:
        video_detect_statistic(args)
    elif args.statistic_filter:
        video_detect_statistic_filter(args)
    else:
        try:
            video_detect_mp(args)
        except KeyboardInterrupt as e:
            print(e)
