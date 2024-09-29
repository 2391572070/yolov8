# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Check a model's accuracy on a test or val split of a dataset.

Usage:
    $ yolo mode=val model=yolov8n.pt data=coco128.yaml imgsz=640

Usage - formats:
    $ yolo mode=val model=yolov8n.pt                 # PyTorch
                          yolov8n.torchscript        # TorchScript
                          yolov8n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                          yolov8n_openvino_model     # OpenVINO
                          yolov8n.engine             # TensorRT
                          yolov8n.mlpackage          # CoreML (macOS-only)
                          yolov8n_saved_model        # TensorFlow SavedModel
                          yolov8n.pb                 # TensorFlow GraphDef
                          yolov8n.tflite             # TensorFlow Lite
                          yolov8n_edgetpu.tflite     # TensorFlow Edge TPU
                          yolov8n_paddle_model       # PaddlePaddle
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode


class BaseValidator:
    """
    BaseValidator.

    A base class for creating validators.

    Attributes:
        args (SimpleNamespace): Configuration for the validator.
        dataloader (DataLoader): Dataloader to use for validation.
        pbar (tqdm): Progress bar to update during validation.
        model (nn.Module): Model to validate.
        data (dict): Data dictionary.
        device (torch.device): Device to use for validation.
        batch_i (int): Current batch index.
        training (bool): Whether the model is in training mode.
        names (dict): Class names.
        seen: Records the number of images seen so far during validation.
        stats: Placeholder for statistics during validation.
        confusion_matrix: Placeholder for a confusion matrix.
        nc: Number of classes.
        iouv: (torch.Tensor): IoU thresholds from 0.50 to 0.95 in spaces of 0.05.
        jdict (dict): Dictionary to store JSON validation results.
        speed (dict): Dictionary with keys 'preprocess', 'inference', 'loss', 'postprocess' and their respective
                      batch processing times in milliseconds.
        save_dir (Path): Directory to save results.
        plots (dict): Dictionary to store plots for visualization.
        callbacks (dict): Dictionary to store various callback functions.
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """
        Initializes a BaseValidator instance.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader to be used for validation.
            save_dir (Path, optional): Directory to save results.
            pbar (tqdm.tqdm): Progress bar for displaying progress.
            args (SimpleNamespace): Configuration for the validator.
            _callbacks (dict): Dictionary to store various callback functions.
        """
        self.args = get_cfg(overrides=args)
        self.dataloader = dataloader
        self.pbar = pbar
        self.stride = None
        self.data = None
        self.device = None
        self.batch_i = None
        self.training = True
        self.names = None
        self.seen = None
        self.stats = None
        self.confusion_matrix = None
        self.nc = None
        self.iouv = None
        self.jdict = None
        self.speed = {'preprocess': 0.0, 'inference': 0.0, 'loss': 0.0, 'postprocess': 0.0}

        self.save_dir = save_dir or get_save_dir(self.args)
        (self.save_dir / 'labels' if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        if self.args.conf is None:
            self.args.conf = 0.001  # default conf=0.001
        self.args.imgsz = check_imgsz(self.args.imgsz, max_dim=1)

        self.plots = {}
        self.callbacks = _callbacks or callbacks.get_default_callbacks()

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """Supports validation of a pre-trained model if passed or a model being trained if trainer is passed (trainer
        gets priority).
        """
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            self.args.half = self.device.type != 'cpu'  # force FP16 val during training
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            # self.model = model
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            callbacks.add_integration_callbacks(self)
            model = AutoBackend(model or self.args.model,
                                device=select_device(self.args.device, self.args.batch),
                                dnn=self.args.dnn,
                                data=self.args.data,
                                fp16=self.args.half)
            # self.model = model
            self.device = model.device  # update device
            self.args.half = model.fp16  # update half
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if engine:
                self.args.batch = model.batch_size
            elif not pt and not jit:
                self.args.batch = 1  # export.py models default to batch-size 1
                LOGGER.info(f'Forcing batch=1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

            if str(self.args.data).split('.')[-1] in ('yaml', 'yml'):
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == 'classify':
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            else:
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

            if self.device.type in ('cpu', 'mps'):
                self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
            if not pt:
                self.args.rect = False
            self.stride = model.stride  # used in get_dataloader() for padding
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

            model.eval()
            model.warmup(imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz))  # warmup



        self.run_callbacks('on_val_start')
        dt = Profile(), Profile(), Profile(), Profile()
        # dt_branch = Profile(), Profile(), Profile(), Profile()
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val

        branches = self.args.branches if isinstance(
            self.args.branches, list) else range(self.args.branches) if isinstance(self.args.branches, int) else []

        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            end_class = 0
            batch_list = []
            last = len(branches) - 1

            batch_branch = batch.copy()

            for i in range(len(branches)):
                batch_list.append({})

            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)


            target = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1),
                             batch['bboxes']), 1)
            # Inference
            with dt[1]:
                for i, branch in enumerate(branches):
                    end_class += branch
                    start_class = end_class
                    start_class -= branch
                    if i == 0:
                        mask = last_mask = target[:, 1] < end_class

                    elif i == last and i > 0:
                        mask = start_class <= target[:, 1]

                    else:
                        _mask = target[:, 1] < end_class
                        mask = last_mask ^ _mask
                        last_mask = _mask

                    # mask = mask.squeeze(dim=1)
                    bat = target[mask]
                    # batch_idx = batch["batch_idx"].to(dtype=torch.long)[mask]
                    idxes_idx = torch.unique(bat[:, 0], sorted=True).to(dtype=torch.long)
                    branch_ID = branches.index(branch)

                    if bat.shape[0] > 1:
                        # idxes_idx = torch.unique(batch_idx, sorted=True).to(dtype=torch.long)
                        batch_img = []
                        batch_img.extend(batch['img'][num] for num in idxes_idx)
                        batch_branch['img'] = torch.stack(batch_img)
                        preds = model(batch_branch['img'], augment=augment)

                        # Loss
                        with dt[2]:
                            batch_im_file, batch_ori_shape, batch_resized_shape, batch_ratio_pad = [],[],[],[]
                            batch_im_file.extend(batch['im_file'][num] for num in idxes_idx)
                            batch_ori_shape.extend(batch['ori_shape'][num] for num in idxes_idx)
                            batch_resized_shape.extend(batch['resized_shape'][num] for num in idxes_idx)
                            batch_ratio_pad.extend(batch['ratio_pad'][num] for num in idxes_idx)

                            batch_branch['im_file'] = list(batch_im_file)
                            batch_branch['ori_shape'] = list(batch_ori_shape)
                            batch_branch['resized_shape'] = list(batch_resized_shape)
                            batch_branch['ratio_pad'] = list(batch_ratio_pad)

                            for j, num in enumerate(bat[:, 0]):
                                bat[:, 0][j] = (idxes_idx == num).nonzero().squeeze(dim=1)

                            if start_class > 0:
                                bat[:, 1] -= start_class

                            batch_branch['batch_idx'], batch_branch['cls'], batch_branch['bboxes'] = bat[:, 0], \
                                            bat[:, 1].view(-1, 1), bat[:, 2:]

                            # idxes_idx = torch.unique(batch_list[i]['batch_idx'], sorted=True).to(dtype=torch.float32)

                            if len(branches) > 1 and self.training:
                                pred = preds[branch_ID][1]
                                pred = [_pred[idxes_idx] for _pred in pred]
                                self.loss += model.loss(batch_branch, pred, branch_ID=branch_ID, idxes_idx=idxes_idx)[1]
                            elif len(branches) == 1 and self.training:
                                self.loss += model.loss(batch_branch, preds[branch_ID][1])[1]

                        # Postprocess
                        with dt[3]:
                            if len(branches) > 1:
                                _preds = self.postprocess(preds[branch_ID])
                            else:
                                _preds = self.postprocess(preds[0])

                        self.update_metrics(_preds, batch_branch, cls_start=start_class)
                        if self.args.plots and batch_i < 3:
                            self.plot_val_samples(batch, batch_i, cls_start=start_class)
                            self.plot_predictions(batch, _preds, batch_i, cls_start=start_class)
                    else:
                        if len(branches) > 1 and self.training:
                            pred = preds[branch_ID][1]
                            pred = [_pred[idxes_idx] for _pred in pred]
                            self.loss[0] += pred[0].sum() * 0
                            self.loss[1] += pred[1].sum() * 0
                            self.loss[2] += pred[2].sum() * 0

            self.run_callbacks("on_val_batch_end")

        stats = self.get_stats()
        self.check_stats(stats)
        self.finalize_metrics()
        self.print_results()
        speed_1 = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1E3 for x in dt)))
        # speed_2 = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1E3 for x in dt_branch)))
        self.run_callbacks('on_val_end')
        if self.training:
            model.float()
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix='val')}
            return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
        else:
            LOGGER.info('Speed: %.1fms preprocess, %.1fms inference, %.1fms loss, %.1fms postprocess per image' %
                        tuple(speed_1.values()))
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / 'predictions.json'), 'w') as f:
                    LOGGER.info(f'Saving {f.name}...')
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")

        return stats

    def match_predictions(self, pred_classes, true_classes, iou, use_scipy=False):
        """
        Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape(N,).
            true_classes (torch.Tensor): Target class indices of shape(M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape(N,10) for 10 IoU thresholds.
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            if use_scipy:
                # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
                import scipy  # scope import to avoid importing for all commands
                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True)
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
            else:
                matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        # matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

    def add_callback(self, event: str, callback):
        """Appends the given callback."""
        self.callbacks[event].append(callback)

    def run_callbacks(self, event: str):
        """Runs all callbacks associated with a specified event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def get_dataloader(self, dataset_path, batch_size):
        """Get data loader from dataset path and batch size."""
        raise NotImplementedError('get_dataloader function not implemented for this validator')

    def build_dataset(self, img_path):
        """Build dataset."""
        raise NotImplementedError('build_dataset function not implemented in validator')

    def preprocess(self, batch):
        """Preprocesses an input batch."""
        return batch

    def postprocess(self, preds):
        """Describes and summarizes the purpose of 'postprocess()' but no details mentioned."""
        return preds

    def init_metrics(self, model):
        """Initialize performance metrics for the YOLO model."""
        pass

    def update_metrics(self, preds, batch, cls_start):
        """Updates metrics based on predictions and batch."""
        pass

    def finalize_metrics(self, *args, **kwargs):
        """Finalizes and returns all metrics."""
        pass

    def get_stats(self):
        """Returns statistics about the model's performance."""
        return {}

    def check_stats(self, stats):
        """Checks statistics."""
        pass

    def print_results(self):
        """Prints the results of the model's predictions."""
        pass

    def get_desc(self):
        """Get description of the YOLO model."""
        pass

    @property
    def metric_keys(self):
        """Returns the metric keys used in YOLO training/validation."""
        return []

    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks)"""
        self.plots[Path(name)] = {'data': data, 'timestamp': time.time()}

    # TODO: may need to put these following functions into callback
    def plot_val_samples(self, batch, ni, cls_start):
        """Plots validation samples during training."""
        pass

    def plot_predictions(self, batch, preds, ni, cls_start):
        """Plots YOLO model predictions on batch images."""
        pass

    def pred_to_json(self, preds, batch):
        """Convert predictions to JSON format."""
        pass

    def eval_json(self, stats):
        """Evaluate and return JSON format of prediction statistics."""
        pass
