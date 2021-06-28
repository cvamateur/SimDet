import torch
import torch.nn as nn

from src.model import FeatureExtractor, PredictionNetworkYOLO
from src.loss import loss_cls_ce, loss_bbox_regression, loss_conf_scores_regression
from utils.bbox_utils import (get_anchor_shapes, generate_grids, generate_anchors, generate_proposals, iou,
                              reference_on_positive_anchors_yolo, nms_fast)
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

    def inference(self, images, thresh=0.5, nms_thresh=0.7):
        """
        Inference-time forward pass for single stage detector.

        @Params:
        -------
        images (tensor):
            Batch of input images of shape [B, 3, 224, 224].
        thresh (float):
            Threshold value on confidence scores. Proposals with conf scores which are less
            than `thresh` will be filtered out.
        nms_thresh (float):
            Threshold on NMS.

        @Returns:
        final_proposals (List[tensor]):
            A list of final proposals, of shape [N, 4], after confidence score thresholding and NMS.
        final_conf_scores (List[tensor]):
            A list of corresponding confidence scores, of shape [N].
        final_classes (List[tensor]):
            A list of corresponding predicted classes, of shape [N].
        """
        final_proposals, final_conf_scores, final_classes = [], [], []

        with torch.no_grad():

            # 1. Extract features
            features = self.feature_extractor(images)
            B, _, h_amap, w_amap = features.shape
            A = self.anc_shapes.shape[0]

            # 2. Predict conf_scores, offsets, cls_scores
            conf_scores, offsets, cls_scores = self.pred_network(features)

            # 3. Apply offsets to anchors, get predicted proposals
            offsets = offsets.permute(0, 1, 3, 4, 2).contiguous()   # [B, A, 4, H', W'] -> [B, A, H', W', 4]
            grids = generate_grids(B, h_amap, w_amap, device=offsets.device)
            anchors = generate_anchors(self.anchor_list.to(grids.device), grids)
            proposals = generate_proposals(anchors, offsets, method="YOLO")  # [B, A, H', W', 4]

            # 4. Get classes
            cls_scores = cls_scores.permute(0, 2, 3, 1).contiguous().unsqueeze(1)
            cls_scores = cls_scores.repeat(1, A, 1, 1, 1)   # [B, A, H', W', C]
            classes = torch.argmax(cls_scores, dim=-1)      # [B, A, H', W']

            # 5. Loop over batch
            for i in range(B):
                batch_proposals = proposals[i].view(-1, 4)
                batch_conf_scores = conf_scores[i].view(-1)
                batch_classes = classes[i].view(-1)

                # confidence thresholding
                conf_mask = batch_conf_scores > thresh
                pick_idxs = torch.nonzero(conf_mask).squeeze(-1)

                picked_proposals = batch_proposals[pick_idxs]
                picked_conf_scores = batch_conf_scores[pick_idxs]
                picked_classes = batch_classes[pick_idxs]

                # NMS thresholding
                nms_idxs = nms_fast(picked_proposals, picked_conf_scores, nms_thresh)

                final_proposals.append(picked_proposals[nms_idxs])
                final_conf_scores.append(picked_conf_scores[nms_idxs])
                final_classes.append(picked_classes[nms_idxs])

        return final_proposals, final_conf_scores, final_classes
