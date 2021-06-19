import torch
import torch.nn as nn

from src.model import FeatureExtractor, PredictionNetworkYOLO
from src.loss import loss_cls_ce, loss_bbox_regression, loss_conf_scores_regression
from utils.bbox_utils import (get_anchor_shapes, generate_grids, generate_anchors, generate_proposals, iou,
                              reference_on_positive_anchors_yolo)
from config import Configs


class SingleStageDetector(torch.nn.Module):
    def __init__(self, cfg):
        super(SingleStageDetector, self).__init__()

        self.cfg = cfg
        self.anc_shapes = get_anchor_shapes(cfg)
        self.feature_extractor = FeatureExtractor(cfg)
        self.predict_head = PredictionNetworkYOLO(cfg.features_dim,
                                                  num_anchors=cfg.num_anchors,
                                                  num_classes=cfg.num_classes)

    def forward(self, images, gt_bboxes):
        """
        Forward pass for single-stage detector for training.

        @Params:
        -------
        images (tensor):
            Batch of input images of shape [B, 3, 224, 224].
        gt_bboxes (tensor):
            GT bounding (padded) boxes of shape [B, N, 5].

        @Returns:
        -------
        total_loss (scalar tensor):
            Total loss for the batch.
        """
        # 1. Get image features
        features = self.feature_extractor(images)

        B, _, h_amap, w_amap = features.shape

        # 2. Get positive and negative anchors
        grids = generate_grids(B, h_amap, w_amap, device=self.cfg.device)
        anchors = generate_anchors(self.anc_shapes, grids)
        iou_mat = iou(anchors, gt_bboxes)
        pos_anc_idx, neg_anc_idx, gt_conf_scores, gt_offsets, gt_cls_ids, *_ = \
            reference_on_positive_anchors_yolo(anchors, gt_bboxes, grids, iou_mat, self.cfg.neg_thresh)

        # 3. Run prediction
        conf_scores, offsets, cls_scores = self.predict_head(features, pos_anc_idx, neg_anc_idx)

        # 4. Calc losses
        loss_conf = loss_conf_scores_regression(conf_scores, gt_conf_scores)
        loss_bbox = loss_bbox_regression(offsets, gt_offsets)
        loss_cls = loss_cls_ce(cls_scores, gt_cls_ids, B, self.cfg.num_anchors * h_amap * w_amap, pos_anc_idx)
        total_loss = self.cfg.w_conf * loss_conf + self.cfg.w_reg * loss_bbox + self.cfg.w_cls * loss_cls
        return total_loss

