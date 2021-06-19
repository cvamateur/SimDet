import torch
import torch.nn as nn

from src.model import FeatureExtractor, PredictionNetworkYOLO
from src.loss import loss_cls_ce, loss_bbox_regression, loss_conf_scores_regression
from utils.bbox_utils import get_anchor_shapes
from config import Configs


class SingleStageDetector(torch.nn.Module):
    def __init__(self, cfg):
        super(SingleStageDetector, self).__init__()

        self.anc_shapes = get_anchor_shapes(cfg)
        self.num_anchors = cfg.num_anchors
        self.num_classes = cfg.num_classes
        self.w_conf = cfg.w_conf
        self.w_reg = cfg.w_reg
        self.w_cls = cfg.w_cls
        self.feature_extractor = FeatureExtractor(cfg)
        self.predictor = PredictionNetworkYOLO(cfg.features_dim,
                                               num_anchors=self.num_anchors,
                                               num_classes=self.num_classes)

    def forward(self, images, gt_bboxes=None):
        """
        Forward pass for single-stage detector. Handle both training and inference.


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
        if gt_bboxes is None:
            return self.inference(images)




    def inference(self, images):
        pass
