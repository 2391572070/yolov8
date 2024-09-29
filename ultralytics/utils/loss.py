# Ultralytics YOLO 🚀, AGPL-3.0 license
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
from ultralytics.nn.modules import SidaDetect, SidaDetectMerge, Detect

from .metrics import bbox_iou
from .tal import bbox2dist


class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction='none') *
                    weight).mean(1).sum()
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self, ):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """Return sum of left and right DFL losses."""
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]) ** 2 + (pred_kpts[..., 1] - gt_kpts[..., 1]) ** 2
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / (2 * self.sigmas) ** 2 / (area + 1e-9) / 2  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()

class SidaDetectionMergeLoss:
    """Criterion class for computing training losses."""
    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters
        self.mod = model
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.nc_branchs = model.model[-1].nc_branchs
        self.device = device
        self.reg_max = 16
        self.use_dfl = self.reg_max > 1
        self.bbox_loss = BboxLoss(self.reg_max - 1, use_dfl=self.use_dfl).to(self.device)
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=self.device)
        # self.stride = m.stride  # model strides

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            if self.is_sida_detect:
                pred_dist = (pred_dist.view(b, a, c // 4, 4).sigmoid() * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
            else:
                pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
                # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
                # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def call_val_loss(self, preds, batch ,branch_size, branch_ID):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        mo = self.mod.model[branch_ID-4]  # Detect() module
        stride = mo.stride[:3]

        self.is_sida_detect = isinstance(mo, (SidaDetect, SidaDetectMerge))

        nc = branch_size
        no = nc + self.reg_max * 4
        assigner = TaskAlignedAssigner(topk=10, num_classes=nc, alpha=0.5, beta=6.0)

        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, stride, 0.5)

        # Targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
        loss *= self.hyp.cls  # cls gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


    def __call__(self, preds, batch, branch_size=None, branch_ID=None):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        if branch_size is not None:
            return self.call_val_loss(preds, batch, branch_size, branch_ID)
        else:
            keys = batch.keys()
            object_batch, coco_batch = {}, {}

            im_file, ori_shape, resized_shape, img, cls, bboxes, batch_idx = [], [], [], [], [], [], []
            im_file1, ori_shape1, resized_shape1, img1, cls1, bboxes1, batch_idx1 = [], [], [], [], [], [], []

            im_file_idx_list_object, im_file_idx_list_coco = [], []

            pred_scores_tl, target_scores_tl, dtype_tl, target_scores_sum_tl = [], [], [], []
            fg_mask_tl, target_bboxes_tl, stride_tensor_tl, pred_distri_tl, \
                pred_bboxes_tl, anchor_points_tl, batch_size_tl = [], [], [], [], [], [], []

            for i in range(len(batch['batch_idx'])):
                if batch['cls'][i].to(self.device) >= torch.tensor([self.nc_branchs[0]]).to(self.device):
                    im_file_idx_list_object.append(int(batch['batch_idx'][i]))
                    cls.append((batch['cls'][i]-self.nc_branchs[0]))
                    bboxes.append(batch['bboxes'][i])
                    batch_idx.append(batch['batch_idx'][i])
                else:
                    im_file_idx_list_coco.append(int(batch['batch_idx'][i]))
                    cls1.append(batch['cls'][i])
                    bboxes1.append(batch['bboxes'][i])
                    batch_idx1.append(batch['batch_idx'][i])

            file_list_object =sorted(set(im_file_idx_list_object))
            file_list_coco = sorted(set(im_file_idx_list_coco))

            for j, k in enumerate(keys):
                if k == 'im_file':
                    im_file.extend((batch['im_file'][num]) for num in list(file_list_object))
                    im_file1.extend((batch['im_file'][num]) for num in list(file_list_coco))
                if k == 'ori_shape':
                    ori_shape.extend((batch['ori_shape'][num]) for num in list(file_list_object))
                    ori_shape1.extend((batch['ori_shape'][num]) for num in list(file_list_coco))
                if k == 'resized_shape':
                    resized_shape.extend((batch['resized_shape'][num]) for num in list(file_list_object))
                    resized_shape1.extend((batch['resized_shape'][num]) for num in list(file_list_coco))
                if k == 'img':
                    img.extend((batch['img'][num]) for num in list(file_list_object))
                    img1.extend((batch['img'][num]) for num in list(file_list_coco))

            if not cls:
                object_batch = []
            else:
                object_batch['im_file'] = tuple(im_file)
                object_batch['ori_shape'] = tuple(ori_shape)
                object_batch['resized_shape'] = tuple(resized_shape)
                object_batch['img'] = torch.stack(img)
                object_batch['cls'] = torch.stack(cls)
                object_batch['bboxes'] = torch.stack(bboxes)
                object_batch['batch_idx'] = torch.reshape(torch.stack(batch_idx), (-1,))

                for i in range(len(object_batch['batch_idx'])):
                    for j in range(len(file_list_object)):
                        if object_batch['batch_idx'][i] == file_list_object[j]:
                            object_batch['batch_idx'][i] = file_list_object.index(file_list_object[j])

            if not cls1:
                coco_batch = []
            else:
                coco_batch['im_file'] = list(im_file1)
                coco_batch['ori_shape'] = list(ori_shape1)
                coco_batch['resized_shape'] = list(resized_shape1)
                coco_batch['img'] = torch.stack(img1)
                coco_batch['cls'] = torch.stack(cls1)
                coco_batch['bboxes'] = torch.stack(bboxes1)
                coco_batch['batch_idx'] = torch.reshape(torch.stack(batch_idx1), (-1,))

                for i in range(len(coco_batch['batch_idx'])):
                    for j in range(len(file_list_coco)):
                        if coco_batch['batch_idx'][i] == file_list_coco[j]:
                            coco_batch['batch_idx'][i] = file_list_coco.index(file_list_coco[j])

            feats_coco = preds[:3][1] if isinstance(preds, tuple) else preds[:3]
            feats_object = preds[3:][1] if isinstance(preds, tuple) else preds[3:]
            feat = [feats_coco, feats_object]
            batch = [coco_batch, object_batch]

            for p in range(len(self.nc_branchs)):
                if not batch[p]:
                    pred_scores_tl.append(0)
                    target_scores_tl.append(0)
                    dtype_tl.append(0)
                    target_scores_sum_tl.append(0)
                    fg_mask_tl.append(0)
                    target_bboxes_tl.append(0)
                    stride_tensor_tl.append(0)
                    pred_distri_tl.append(0)
                    pred_bboxes_tl.append(0)
                    anchor_points_tl.append(0)
                    batch_size_tl.append(0)
                    continue
                else:
                    m = self.mod.model[p-3]  # Detect() module
                    stride = m.stride[:3] #################################################
                    nc = m.nc  # number of classes
                    no = m.no
                    # self.is_sida_detect = isinstance(m, (SidaDetect, SidaDetectMerge))
                    feats = feat[p]

                    if self.nc_branchs[p] == 12:
                        for j in range(len(feats)):
                            feats[j] = feats[j][file_list_coco]
                    else:
                        for j in range(len(feats)):
                            feats[j] = feats[j][file_list_object]


                    pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], no, -1) for xi in feats], 2).split(
                                                        (self.reg_max * 4, nc), 1)

                    pred_scores = pred_scores.permute(0, 2, 1).contiguous()
                    pred_distri = pred_distri.permute(0, 2, 1).contiguous()

                    dtype = pred_scores.dtype
                    batch_size = pred_scores.shape[0]
                    imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * stride[0]  # image size (h,w)

                    anchor_points, stride_tensor = make_anchors(feats, stride, 0.5)

                    # Targets
                    targets = torch.cat((batch[p]['batch_idx'].view(-1, 1), batch[p]['cls'].view(-1, 1), batch[p]['bboxes']), 1)
                    targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
                    gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
                    mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

                    # Pboxes
                    pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

                    assigner = TaskAlignedAssigner(topk=10, num_classes=nc, alpha=0.5, beta=6.0)
                    _, target_bboxes, target_scores, fg_mask, _ = assigner(
                        pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                        anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

                    target_scores_sum = max(target_scores.sum(), 1)

                    pred_scores_tl.append(pred_scores)
                    target_scores_tl.append(target_scores)
                    dtype_tl.append(dtype)
                    target_scores_sum_tl.append(target_scores_sum)
                    fg_mask_tl.append(fg_mask)
                    target_bboxes_tl.append(target_bboxes)
                    stride_tensor_tl.append(stride_tensor)
                    pred_distri_tl.append(pred_distri)
                    pred_bboxes_tl.append(pred_bboxes)
                    anchor_points_tl.append(anchor_points)
                    batch_size_tl.append(batch_size)

            batch_size_sum = sum(batch_size_tl)
            loss = torch.zeros(3, device=self.device)  # box, cls, dfl
            for i in range(len(self.nc_branchs)):
                if not batch[i]:
                    loss_empty = [feat[i][0].sum() * 0, feat[i][1].sum() * 0, feat[i][2].sum() * 0]
                    loss[0] += loss_empty[0]
                    loss[1] += loss_empty[1]
                    loss[2] += loss_empty[2]
                else:
                    # Cls loss
                    # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
                    # batch_size_sum = sum(batch_size_tl)
                    loss1 = self.bce(pred_scores_tl[i], target_scores_tl[i].to(dtype_tl[i])).sum() / target_scores_sum_tl[i]  # BCE
                    loss[1] += loss1
                    # Bbox loss
                    if fg_mask_tl[i].sum():
                        target_bboxes_tl[i] /= stride_tensor_tl[i]
                        loss0, loss2 = self.bbox_loss(pred_distri_tl[i], pred_bboxes_tl[i], anchor_points_tl[i], target_bboxes_tl[i],
                                                          target_scores_tl[i], target_scores_sum_tl[i], fg_mask_tl[i])
                        loss[0] += loss0
                        loss[2] += loss2

            loss[0] *= self.hyp.box  # box gain
            loss[1] *= self.hyp.cls  # cls gain
            loss[2] *= self.hyp.dfl  # dfl gain

            return loss.sum() * batch_size_sum, loss.detach()  # loss(box, cls, dfl)


class SidaDetectionMergeLossV1:
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters
        self.mod = model
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.device = device
        self.reg_max = 16
        self.use_dfl = self.reg_max > 1
        self.bbox_loss = BboxLoss(self.reg_max - 1, use_dfl=self.use_dfl).to(self.device)
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            if self.is_sida_detect:
                pred_dist = (pred_dist.view(b, a, c // 4, 4).sigmoid() * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
            else:
                pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
                # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
                # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def call_val_loss(self, preds, batch, branch_ID, idxes_idx):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)
        loss_weight = 1
        branches = self.hyp.branches if isinstance(
            self.hyp.branches, list) else range(self.hyp.branches) if isinstance(self.hyp.branches, int) else []

        if isinstance(self.mod.model[-1], Detect):
            mo = self.mod.model[-1]
        else:
            mo = self.mod.model[branch_ID - len(branches)-1]
            loss_weight_list = [0.95, 0.05]
            loss_weight = loss_weight_list[branch_ID-1]

        feats = preds[1] if isinstance(preds, tuple) else preds
        # Targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)

        stride = mo.stride[:3]
        self.is_sida_detect = isinstance(mo, (SidaDetect, SidaDetectMerge))
        nc = mo.nc
        no = nc + self.reg_max * 4
        assigner = TaskAlignedAssigner(topk=10, num_classes=nc, alpha=0.5, beta=6.0)

        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]

        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, stride, 0.5)

        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size * loss_weight, loss.detach() * loss_weight  # loss(box, cls, dfl)

    def __call__(self, preds, batch, branch_ID=None, idxes_idx_val=None):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        if branch_ID is not None:
            return self.call_val_loss(preds, batch, branch_ID, idxes_idx_val)
        else:
            branches = self.hyp.branches if isinstance(
                self.hyp.branches, list) else range(self.hyp.branches) if isinstance(self.hyp.branches, int) else []

            loss = torch.zeros(3, device=self.device)
            Loss = torch.zeros(3, device=self.device)# box, cls, dfl
            preds_count = 0
            end_class = 0
            start_class = 0
            last = len(branches) - 1
            batch_size_list = []
            loss_list = []
            remain_idx = None
            # Targets
            targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)

            for i in range(len(branches)):
                if isinstance(self.mod.model[-1], Detect):
                    det = self.mod.model[-1]
                    pred = preds
                else:
                    det = self.mod.model[i-len(branches)-1]
                    pred = preds[preds_count:preds_count + 3]
                    preds_count += 3

                self.is_sida_detect = isinstance(det, (SidaDetect, SidaDetectMerge))

                _loss = torch.zeros(3, device=self.device)
                no = det.no
                nc = det.nc
                stride = det.stride[:3]

                end_class += nc
                if i == 0:
                    mask = last_mask = targets[:, 1] < end_class
                elif i == last and i > 1:
                    mask = start_class <= targets[:, 1]
                else:
                    _mask = targets[:, 1] < end_class
                    mask = last_mask ^ _mask
                    last_mask = _mask

                target = targets[mask]
                remain_target = targets[~mask]
                # batch_idx = batch["batch_idx"].to(dtype=torch.long)[mask]
                idxes_idx = torch.unique(target[:, 0], sorted=True).to(dtype=torch.long)

                if remain_target.shape[0] > 0:
                    remain_idx = torch.unique(remain_target[:, 0], sorted=True).to(dtype=torch.long)

                batch_size = len(idxes_idx)
                batch_size_list.append(batch_size)

                if target.shape[0] < 1:
                    loss_empty = [pred[0].sum() * 0, pred[1].sum() * 0, pred[2].sum() * 0]
                    _loss[0] = loss_empty[0]
                    _loss[1] = loss_empty[1]
                    _loss[2] = loss_empty[2]

                else:
                    feats = pred[1] if isinstance(pred, tuple) else pred
                    remain_feats = feats.copy()
                    feats = [feat[idxes_idx] for feat in feats]

                    pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], no, -1) for xi in feats], 2).split(
                        (self.reg_max * 4, nc), 1)

                    pred_scores = pred_scores.permute(0, 2, 1).contiguous()
                    pred_distri = pred_distri.permute(0, 2, 1).contiguous()

                    dtype = pred_scores.dtype

                    imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * stride[0]  # image size (h,w)
                    anchor_points, stride_tensor = make_anchors(feats, stride, 0.5)

                    idxes_idx = idxes_idx.to(dtype=torch.float32)
                    for j, num in enumerate(target[:, 0]):
                        target[:, 0][j] = (idxes_idx == num).nonzero().squeeze(dim=1)

                    if start_class > 0:
                        target[:, 1] -= start_class
                    target = self.preprocess(target.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
                    gt_labels, gt_bboxes = target.split((1, 4), 2)  # cls, xyxy
                    mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

                    # Pboxes
                    pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

                    assigner = TaskAlignedAssigner(topk=10, num_classes=nc, alpha=0.5, beta=6.0)
                    _, target_bboxes, target_scores, fg_mask, _ = assigner(
                        pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                        anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

                    target_scores_sum = max(target_scores.sum(), 1)

                    # Cls loss
                    # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
                    loss1 = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum   # BCE
                    # loss[1] += loss1
                    _loss[1] = loss1

                    # Bbox loss
                    if fg_mask.sum():
                        target_bboxes /= stride_tensor
                        loss0, loss2 = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                                          target_scores_sum, fg_mask)
                        _loss[0] = loss0
                        _loss[2] = loss2

                    if remain_idx is not None:
                        remain_feats = [feat[remain_idx] for feat in remain_feats]
                        loss_empty = [remain_feats[0].sum() * 0, remain_feats[1].sum() * 0, remain_feats[2].sum() * 0]
                        _loss[0] += loss_empty[0]
                        _loss[1] += loss_empty[1]
                        _loss[2] += loss_empty[2]

                start_class = end_class
                loss_list.append(_loss)

            loss_weight = [0.95, 0.05]
            for i in range(len(branches)):
                loss[0] += loss_list[i][0] * self.hyp.box * batch_size_list[i] * loss_weight[i]
                loss[1] += loss_list[i][1] * self.hyp.cls * batch_size_list[i] * loss_weight[i]
                loss[2] += loss_list[i][2] * self.hyp.dfl * batch_size_list[i] * loss_weight[i]

                Loss[0] += loss_list[i][0] * self.hyp.box * loss_weight[i]
                Loss[1] += loss_list[i][1] * self.hyp.cls * loss_weight[i]
                Loss[2] += loss_list[i][2] * self.hyp.dfl * loss_weight[i]

            return loss.sum(), Loss.detach()  # loss(box, cls, dfl)




class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()


        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)

        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes the v8SegmentationLoss class, taking a de-paralleled model as argument."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch['batch_idx'].view(-1, 1)
            targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError('ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n'
                            "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                            "i.e. 'yolo train model=yolov8n-seg.pt data=coco128.yaml'.\nVerify your dataset is a "
                            "correctly formatted 'segment' dataset using 'data=coco128-seg.yaml' "
                            'as an example.\nSee https://docs.ultralytics.com/tasks/segment/ for help.') from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
                                              target_scores, target_scores_sum, fg_mask)
            # Masks loss
            masks = batch['masks'].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode='nearest')[0]

            loss[1] = self.calculate_segmentation_loss(fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto,
                                                       pred_masks, imgsz, self.overlap)

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def single_mask_loss(gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor,
                         area: torch.Tensor) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction='none')
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i],
                                              marea_i[fg_mask_i])

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch['batch_idx'].view(-1, 1)
        targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)
            keypoints = batch['keypoints'].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(fg_mask, target_gt_idx, keypoints, batch_idx,
                                                             stride_tensor, target_bboxes, pred_kpts)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes,
                                 pred_kpts):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            (tuple): Returns a tuple containing:
                - kpts_loss (torch.Tensor): The keypoints loss.
                - kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros((batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]),
                                        device=keypoints.device)

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, :keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2]))

        # Divide coordinates by stride
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        loss = torch.nn.functional.cross_entropy(preds, batch['cls'], reduction='mean')
        loss_items = loss.detach()
        return loss, loss_items
