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
import sys
import torch
import torch.nn.functional as F
from collections import OrderedDict
import multiprocessing
import random

sys.path.append('.')

seq_width = 28
seq_height = 28


def parse_args():
    parser = argparse.ArgumentParser(description='Image Detect')
    parser.add_argument('--base_path', type=str, required=True, help='image base path')
    parser.add_argument('--list_file', type=str, required=True, help='image list file')
    parser.add_argument('--out_path', type=str, required=True, help='out path')
    parser.add_argument('--checkpoint', type=str, required=True, help='checkpoint file path')
    parser.add_argument('--proccess_count', type=int, default=16, help='proccess count')
    parser.add_argument('--gpu_count', type=int, default=1, help='gpu count')
    parser.add_argument('--gpu_id_offset', type=int, default=0, help='gpu id offset')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='inference size h,w')
    parser.add_argument("--stride", type=int, default=32, help="image stride")
    parser.add_argument("--auto", action='store_true', help="auto")
    parser.add_argument('--conf_thres', type=float, default=0.1, help='confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max_det', type=int, default=300, help='maximum detections per image')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic_nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--multi_label', action='store_true', help='multi label')
    parser.add_argument('--min_area', type=int, default=0, help='Bbox min area')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--name', type=str, default='', help='out name')
    parser.add_argument("--feat_mode", type=int, default=1, help='save feat mode, 0 is all feat, 1 is only class feat, 2 is only fuse label feat, 3 is only fuse label feat but large feat map on toper')
    parser.add_argument("--save_feat", action='store_true', help="save feat")
    parser.add_argument("--display", action='store_true', help="display detect results")
    parser.add_argument("--display_feat", action='store_true', help="display feat")
    parser.add_argument("--display_fuse_feat", action='store_true', help="display fuse feat")
    parser.add_argument('--display_fuse_mode', type=int, default=0, help='display fuse mode')
    parser.add_argument("--display_feat_use_softmax", action='store_true', help="display feat use softmax")
    parser.add_argument("--pause", action='store_true', help="pause display detect results")
    parser.add_argument('--seq_width', type=int, default=28, help='seq width')
    parser.add_argument('--seq_height', type=int, default=28, help='seq height')
    args = parser.parse_args()
    return args


args = parse_args()

random.seed(12345)


def os_system(cmd_str):
    if sys.platform.startswith('linux'):
        os.system(cmd_str)


def process_detect_func(results, filenames, score_thr, min_area, feat_mode):
    results, pred_feats, feat_shapes = results
    bbox_count = 0
    filename_list = []
    bboxes_list = []
    for result, filename in zip(results, filenames):
        bboxes = result.cpu().numpy()

        out_bboxes = []
        if bboxes.size > 0:
            mask = bboxes[..., 4] >= score_thr
            bboxes = bboxes[mask]
            if bboxes.size > 0:
                # scores = bboxes[..., 4]
                if min_area > 0:
                    area = np.prod(bboxes[..., 2:4] - bboxes[..., :2], 1)
                    vaild_mask = area > min_area
                    # scores = scores[vaild_mask]
                    bboxes = bboxes[vaild_mask]

                out_bboxes = bboxes.tolist()
                bbox_count += len(out_bboxes)

        filename_list.append(filename)
        bboxes_list.append(out_bboxes)

    results_dict = OrderedDict()
    results_dict['filename'] = filename_list
    results_dict['bbox'] = bboxes_list
    results_dict['seg'] = None
    results_dict['frame_count'] = len(filename_list)
    if pred_feats is not None:
        if 1 == feat_mode:
            pred_feats = pred_feats[:, 4:, ...]
        elif 1 < feat_mode:
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


def detect_run(file_queue, out_queue, pid, args):
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

                (images, filenames, w_hs), _, _ = file_bboxes

                if images is not None:
                    # images.requires_grad = False
                    # images = images.cuda(non_blocking=True)
                    results = model(images, w_hs)
                    results = process_detect_func(results, filenames, args.conf_thres, args.min_area, args.feat_mode)
                else:
                    results = None

                while out_queue.qsize() > 100:
                    time.sleep(0.01)
                out_queue.put(((results, ), False, pid))
    except Exception as e:
        if str(e) != '':
            print('detect_run', e)

    out_queue.put((None, True, pid))


def image_run(file_queue, out_queue, pid, args):
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

        def put_images(out_queue, images, filenames, w_hs):
            while out_queue.qsize() > 64:
                time.sleep(0.01)
            out_queue.put(((images, filenames, w_hs), False, pid))

        def pre_batch_image(out_queue, batch_images, batch_filenames, batch_w_hs):
            image_count = len(batch_images)
            if image_count > 0:
                input_images = torch.stack(batch_images, 0)
                input_images.share_memory_()
                input_filenames = batch_filenames.copy()
                input_w_hs = batch_w_hs.copy()
                put_images(out_queue, input_images, input_filenames, input_w_hs)
                batch_images.clear()
                batch_filenames.clear()
                batch_w_hs.clear()

        batch_size = args.batch_size
        detect_trans = DetectTransform(args)
        batch_filenames = []
        batch_images = []
        batch_w_hs = []
        frame_count = 0
        while True:
            file_bboxes = file_queue.get(block=True)
            if file_bboxes is None:
                break

            file_name, image_path = file_bboxes

            image_bgr = cv2.imread(image_path)
            frame_count += 1

            # image_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            image = detect_trans(image_bgr)
            batch_images.append(image)
            batch_filenames.append(file_name)
            batch_w_hs.append([image_bgr.shape[1], image_bgr.shape[0]])
            batch_image_count = len(batch_images)
            if batch_image_count >= batch_size:
                pre_batch_image(out_queue, batch_images, batch_filenames, batch_w_hs)
    except Exception as e:
        if str(e) != '':
            print('image_run', e)

    pre_batch_image(out_queue, batch_images, batch_filenames, batch_w_hs)


def out_run(out_queue, args, out_file_path, total_file_count):
    out_dir = os.path.dirname(out_file_path)
    out_file = open(out_file_path, 'a+', encoding='utf-8')
    out_base_name = os.path.splitext(out_file_path)[0]
    save_feat = args.save_feat
    if save_feat:
        out_feat_file_path = out_base_name + '.feat.dat'
        out_feat_file = open(out_feat_file_path, 'wb')

    file_count = 0
    batch_count = 0
    try:
        finish_worker_count = 0
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
                _results, = results
                if _results is not None:
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
                        print('{} {}/{} {} {} Process Id {}: FPS {:.3f} {:.3f}'.format((end_time - begin_time), file_count, total_file_count, total_frame_count, _total_frame_count, pid, fps1, fps2))

                    filenames = results_dict.pop('filename', None)
                    bboxes = results_dict.pop('bbox', None)
                    feats = results_dict.pop('feat', None)
                    segs = results_dict.pop('seg', None)

                    if (filenames is None) or (bboxes is None):
                        continue

                    for filename, _bboxes in zip(filenames, bboxes):
                        if len(_bboxes) > 0:
                            bbox_strs = []
                            for bbox in _bboxes:
                                # label,x1,y1,x2,y2,score
                                bbox_str = '{},{},{},{},{},{:.4f}'.format(int(bbox[5]), int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), bbox[4])
                                bbox_strs.append(bbox_str)
                            out_info = '{} {}\n'.format(filename, ';'.join(bbox_strs))
                        else:
                            out_info = '{}\n'.format(filename)
                        out_file.write(out_info)
                    file_count += len(filenames)

                    if save_feat and (feats is not None):
                        feats = feats.numpy()
                        # print(feats.dtype, feats.shape)
                        feats.tofile(out_feat_file)

    except Exception as e:
        print('out_run', e)
    out_file.close()
    os_system('chmod a+wr \"{}\" -f'.format(out_file_path))
    if save_feat:
        out_feat_file.close()
        os_system('chmod a+wr \"{}\" -f'.format(out_feat_file_path))


def image_detect_mp(args):
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
        os_system('chmod a+wr \"{}\"'.format(args.out_path))
    manager = multiprocessing.Manager()
    file_queue = manager.Queue()
    out_queue = manager.Queue()
    image_queue = manager.Queue()

    out_file_path = os.path.join(args.out_path, 'image_detect_bbox.txt')
    filter_file_names = set()
    if os.path.exists(out_file_path):
        try:
            with open(out_file_path, 'r', encoding='utf-8') as file:
                for line in file.readlines():
                    lines = line.strip().split()
                    file_name = lines[0]
                    filter_file_names.add(file_name)
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
        workers.append(multiprocessing.Process(target=image_run, args=(file_queue, image_queue, i, args)))

    det_workers = []
    for i in range(args.gpu_count):
        det_workers.append(multiprocessing.Process(target=detect_run, args=(image_queue, out_queue, i, args)))

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
        image_path = os.path.join(video_dir, file_name)
        if not os.path.exists(image_path):
            continue
        file_queue.put((file_name, image_path))
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

    # detect_run(file_queue, out_queue, 0, args, file_lists[0])
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


def image_detect_display(args):
    from ultralytics.yolo.utils.plotting import Colors

    opacity = 0.5

    out_file_path = os.path.join(args.out_path, 'image_detect_bbox.txt')

    if args.classes is not None:
        classes = set(args.classes)
        if len(classes) < 1:
            classes = None
    else:
        classes = None

    label_names = set()
    if args.checkpoint is not None:
        if os.path.exists(args.checkpoint):
            with torch.no_grad():
                with open(args.checkpoint, 'rb') as file:
                    model = torch.load(file, map_location='cpu')
                # from ultralytics.nn.autobackend import AutoBackend
                # model = AutoBackend(args.checkpoint)
                model_info = model.get('model', None)
                if model_info is not None:
                    label_names = model_info.names

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

    padding = Padding(args)

    # color_list = []
    color_map = {}

    colors = Colors()
    color_tensor = None

    main_win_name = 'image detect'
    cv2.namedWindow(main_win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    video_dir = args.base_path
    out_dir = os.path.dirname(out_file_path)

    file_base_name = os.path.splitext(out_file_path)[0]
    seq_file = None

    # if display_feat:
    #     feat_file_path = os.path.join(out_dir, file_base_name + '.feat.dat')
    #     if os.path.exists(feat_file_path):
    #         feat_infos = np.load(feat_file_path)
    #         frameids = feat_infos['frameids']
    #         feats = feat_infos['feats']
    #         feat_infos = {frameid: feat for frameid, feat in zip(frameids, feats)}
    #     else:
    #         feat_infos = None
    #
    # height, width = args.imgsz
    #
    # padding_info = padding.cal_padding(width, height)
    # left, top, right, bottom = padding_info[0]
    # ow, oh = padding_info[1]
    # pw, ph = padding_info[2]
    # layer_sizes = []
    # layer_shapes = []
    # align_w = None
    # align_h = None
    # for s in [8, 16, 32]:
    #     h, w = ph // s, pw // s
    #     layer_sizes.append(h*w)
    #     layer_shapes.append((h, w))
    #     if align_h is None:
    #         align_h = h
    #     else:
    #         if align_h < h:
    #             align_h = h
    #     if align_w is None:
    #         align_w = w
    #     else:
    #         if align_w < w:
    #             align_w = w

    frame_count = 0
    with open(out_file_path, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            lines = line.strip().split()
            file_name = lines[0]
            if len(lines) > 1:
                display_info = []
                bboxes = lines[1]
                for bbox in bboxes.split(';'):
                    bbox = bbox.split(',')
                    label = int(bbox[0])
                    x1 = int(bbox[1])
                    y1 = int(bbox[2])
                    x2 = int(bbox[3])
                    y2 = int(bbox[4])
                    score = float(bbox[5])
                    seq = None
                    display_info.append([label, [x1, y1, x2, y2], score, seq])
            else:
                display_info = None

            image_path = os.path.join(video_dir, file_name)
            image_bgr = cv2.imread(image_path)

            # height, width = image_bgr.shape[0:2]
            if display_info is not None:
                for label, bbox, score, seq in display_info:
                    if classes is not None:
                        if label not in classes:
                            continue
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
                    cv2.putText(image_bgr, '{} {:.2f}'.format(label_names.get(label, label), score), (bbox[0], bbox[1]-4), cv2.FONT_HERSHEY_COMPLEX, 1, color, thickness=2)

            frame_count += 1

            # if display_feat:
            #     image_bgr = padding(image_bgr)
            #     if feat_infos is not None:
            #         feat = feat_infos.get(frame_count, None)
            #         if feat is not None:
            #             if feat_mode <= 1:
            #                 if color_tensor is None:
            #                     color_tensor = torch.tensor([colors(i) for i in range(feat.shape[0])], dtype=torch.uint8)
            #                     _colors = color_tensor[:, None].numpy()
            #                 if 1 == feat_mode:
            #                     feat = torch.from_numpy(feat).float()
            #                 else:
            #                     feat = torch.from_numpy(feat[4:, ...]).float()
            #                 if multi_label:
            #                     feats = []
            #                     _feats = feat.split(layer_sizes, -1)
            #                     for i, layer_shape in enumerate(layer_shapes):
            #                         _feat = _feats[i].view(-1, *layer_shape)
            #                         if (layer_shape[0] != align_h) or (layer_shape[1] != align_w):
            #                             # _feat = F.interpolate(_feat[None], size=(align_h, align_w), mode='nearest')[0]
            #                             _feat = F.interpolate(_feat[None], size=(align_h, align_w), mode='bilinear', align_corners=False)[0]
            #                         feats.append(_feat)
            #                     feats = torch.stack(feats, -1)
            #                     feats = torch.max(feats, -1)[0]
            #                     factors = F.interpolate((1-feats)[None], size=(image_bgr.shape[0], image_bgr.shape[1]), mode='bilinear', align_corners=False)[0][:,:,:,None].numpy()
            #                     feats = feats.numpy()
            #
            #                     feat_images = []
            #                     fuse_feat_images = []
            #                     for i, _feat in enumerate(feats):
            #                         if classes is not None:
            #                             if i not in classes:
            #                                 continue
            #                         color = _colors[i]
            #                         feat_image = (_feat[:, :, None] * color).astype(np.uint8)
            #                         feat_images.append(feat_image)
            #                         # feat_image = cv2.resize(feat_image, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            #                         feat_image = cv2.resize(feat_image, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
            #                         fuse_feat_image = (factors[i]*image_bgr + feat_image).astype(np.uint8)
            #                         fuse_feat_images.append(fuse_feat_image)
            #                         cv2.imshow('feat{}'.format(i), feat_image)
            #                         cv2.imshow('fuse{}'.format(i), fuse_feat_image)
            #                 elif display_fuse_feat:
            #                     feats = []
            #                     _feats = feat.split(layer_sizes, -1)
            #                     for i, layer_shape in enumerate(layer_shapes):
            #                         _feat = _feats[i].view(-1, *layer_shape)
            #                         if (layer_shape[0] != align_h) or (layer_shape[1] != align_w):
            #                             # _feat = F.interpolate(_feat[None], size=(align_h, align_w), mode='nearest')[0]
            #                             _feat = F.interpolate(_feat[None], size=(align_h, align_w), mode='bilinear', align_corners=False)[0]
            #                         feats.append(_feat)
            #
            #                     if display_fuse_mode == 0:
            #                         feat_count = len(feats)
            #                         feat_score_labels = [torch.max(f, 0) for f in feats[::-1]]
            #                         feat_score, feat_label = feat_score_labels[0]
            #                         for i in range(1, feat_count):
            #                             _feat_score, _feat_label = feat_score_labels[i]
            #                             same_label_mask = feat_label == _feat_label
            #                             diff_label_mask = torch.logical_not(same_label_mask)
            #                             feat_score[same_label_mask] = torch.maximum(feat_score[same_label_mask], _feat_score[same_label_mask])
            #                             if score_thr > 0:
            #                                 score_mask = _feat_score >= score_thr
            #                                 diff_label_mask.logical_and_(score_mask)
            #                             feat_score[diff_label_mask] = _feat_score[diff_label_mask]
            #                             feat_label[diff_label_mask] = _feat_label[diff_label_mask]
            #                     else:
            #                         feats = torch.stack(feats, -1)
            #                         feats = torch.max(feats, -1)[0]
            #                         feat_score, feat_label = torch.max(feats, 0)
            #
            #                     feat_image = (color_tensor[feat_label] * feat_score[:, :, None])
            #                     if score_thr > 0:
            #                         feat_image[feat_score < score_thr] = 0
            #                     feat_image = feat_image.numpy().astype(np.uint8)
            #                     feat_image = cv2.resize(feat_image, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
            #                     alpha = F.interpolate((1 - feat_score)[None, None], size=(image_bgr.shape[0], image_bgr.shape[1]), mode='bilinear', align_corners=False)[0][0][:, :, None].numpy()
            #                     fuse_feat_image = (alpha * image_bgr + feat_image).astype(np.uint8)
            #                     cv2.imshow('feat', feat_image)
            #                     cv2.imshow('fuse', fuse_feat_image)
            #                 else:
            #                     if display_feat_use_softmax:
            #                         feat = torch.softmax(feat, 0)
            #                     feat_score, feat_label = torch.max(feat, 0)
            #                     # feat_label = torch.argmax(feat, 0)
            #                     feat_images = color_tensor[feat_label]
            #                     if score_thr > 0:
            #                         feat_images[feat_score < score_thr] = 0
            #                     feat_images = feat_images.split(layer_sizes, 0)
            #                     _feat_images = []
            #                     for i, layer_shape in enumerate(layer_shapes):
            #                         feat_image = feat_images[i].view(*layer_shape, -1).numpy()
            #                         if (layer_shape[0] != align_h) or (layer_shape[1] != align_w):
            #                             feat_image = cv2.resize(feat_image, (align_w, align_h), interpolation=cv2.INTER_NEAREST)
            #                             # feat_image = cv2.resize(feat_image, (align_w, align_h), interpolation=cv2.INTER_LINEAR)
            #                             # feat_image = F.interpolate(feat_image, (align_h, align_w, feat_image.shape[-1]), mode='nearest')
            #                             # feat_image = F.interpolate(feat_image, (align_h, align_w, feat_image.shape[-1]), mode='bilinear', align_corners=False)
            #                         _feat_images.append(feat_image)
            #                         cv2.imshow('feat{}'.format(i), feat_image)
            #
            #                     feat_images = _feat_images[::-1]
            #                     fuse_feat_image = None
            #                     for feat_image in feat_images:
            #                         if fuse_feat_image is None:
            #                             fuse_feat_image = feat_image
            #                         else:
            #                             mask = feat_image.sum(axis=-1) > 0
            #                             fuse_feat_image[mask] = feat_image[mask]
            #                     fuse_feat_image = cv2.resize(fuse_feat_image, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            #                     cv2.imshow('fuse feat', fuse_feat_image)
            #                     mask = 0 != fuse_feat_image
            #                     image_bgr[mask] = ((1-opacity)*image_bgr[mask] + opacity*fuse_feat_image[mask]).astype(np.uint8)
            #             elif 1 < feat_mode:
            #                 if color_tensor is None:
            #                     color_tensor = torch.tensor([colors(i) for i in range(128)], dtype=torch.uint8)
            #                 feat = torch.from_numpy(feat)
            #                 feat_label = feat[0].long()
            #                 feat_score = feat[1].float()
            #                 feat_image = color_tensor[feat_label] * feat_score[:, :, None]
            #                 if score_thr > 0:
            #                     feat_image[feat_score < score_thr] = 0
            #                 feat_image = feat_image.numpy().astype(np.uint8)
            #                 feat_image = cv2.resize(feat_image, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
            #                 alpha = F.interpolate((1 - feat_score)[None, None], size=(image_bgr.shape[0], image_bgr.shape[1]), mode='bilinear', align_corners=False)[0][0][:, :, None].numpy()
            #                 fuse_feat_image = (alpha * image_bgr + feat_image).astype(np.uint8)
            #                 cv2.imshow('feat', feat_image)
            #                 cv2.imshow('fuse', fuse_feat_image)
            # else:
            #     image_bgr = cv2.resize(image_bgr, (width // 2, height // 2))
            print(frame_count)
            cv2.imshow(main_win_name, image_bgr)
            if pause:
                key = cv2.waitKey()
            else:
                key = cv2.waitKey(1)
            if key == 27:
                return


if __name__ == "__main__":
    if args.display:
        image_detect_display(args)
    else:
        try:
            image_detect_mp(args)
        except KeyboardInterrupt as e:
            print(e)
