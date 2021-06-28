import torch
from torch import nn
from torchvision import models
from torchsummary import summary


class FeatureExtractor(nn.Module):
    """
    Image feature extraction using MobileNet-V2.
    """
    def __init__(self, cfg, verbose=False):
        super(FeatureExtractor, self).__init__()
        self.input_shape = (3, cfg.input_height, cfg.input_width)
        self.model = models.mobilenet_v2(pretrained=True).features
        if verbose:
            if cfg.device == "cuda":
                self.model.to("cuda")
            summary(self.model, (3, 224, 224))

    def forward(self, image):
        """
        @Params:
        -------
        image (tensor) 4-D tensor input image where the
        """
        return self.model(image)


class PredictionNetworkYOLO(nn.Module):
    """
    Predict classification scores and transformations for each anchor given input features
    from the FeatureExtractor.

    Details:
        For each position in the 7x7 grid of features from the backbone, the prediction network
        outputs `C` numbers to be interpreted as classification scores over `C` object categories
        for the anchors at that position.

        For each of the `A` anchors at that position, the prediction network outputs a transformation
        (tx, ty, tw, th) and a confidence score (large positive values indicate high probability that
        the anchor contains an object).

        Collecting all of these outputs, we see that for each position in the 7x7 grid of features,
        we need to output a total of `5A+C` numbers, that is the prediction work receives an input
        tensor of shape (B, in_dim, 7, 7) and produces an output tensor of shape (B, 5A+C, 7, 7).

        During training, we do not apply the loss on the full set of anchor boxes for the image;
        instead we designate a subset of anchors as positive and negative by matching them with
        ground truth boxes by `box_utils.reference_on_positive_anchors()`. The prediction network
        is also responsible for picking out the outputs corresponding to the pos and neg anchors.
    """
    def __init__(self, in_dim, hidden_dim=128, num_anchors=9, num_classes=20, drop_ratio=0.3):
        super(PredictionNetworkYOLO, self).__init__()
        assert num_anchors != 0 and num_classes != 0
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        # network yo predict outputs
        self.pred_layer = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.Dropout(p=drop_ratio),
            nn.Conv2d(hidden_dim, 5*num_anchors+num_classes, kernel_size=(1, 1), stride=(1, 1)),
        )

    def forward(self, features, pos_anc_idx=None, neg_anc_idx=None):
        """
        Run forward pass of the network to predict outputs given features from the backbone network.

        The outputs are different during training and inference:
        During training:
            `pos_anc_idx` and `neg_anc_idx` are given, and identify which anchors should be positive
            and negative, this forward pass needs to extract only the predictions for the positive
            and negative anchors.

        During inference:
            only features are provided and this method needs to return predictions for all anchors.

        @Params:
        -------
        features (tensor):
            Features from backbone network of shape (B, in_dim, 7, 7).
        pos_anc_idx (tensor):
            int64 tensor of shape [M] giving the indices of anchors marked as positive. Must be
            given during training, and None during inference.
        neg_anc_idx (tensor):
            int64 tensor of shape [M] giving the indices of anchors marked as negative. Must be
            given during training, and None during inference.

        @Returns (During training):
        -------
        conf_scores (tensor):
            Tensor of shape [2*M] giving the predicted confidence scores for positive anchors
            and negative anchors (in that order).
        offsets (tensor):
            Tensor of shape [M, 4] giving predicted transformation for positive anchors.
        cls_scores (tensor):
            Tensor of shape [M, C) giving predicted classification scores for positive anchors.

        @Returns (During inference):
        conf_scores (tensor):
            Tensor of shape [B, A, H', W'] giving predicted confidence scores for all anchors.
        offsets (tensor):
            Tensor of shape [B, A, 4, H', W'] predicted transformations all all anchors.
        cls_scores (tensor):
            Tensor of shape [B, C, H', W'] giving predicted classification scores for all anchors.
        """
        out = self.pred_layer(features)     # [B, 5A+C, H', W']

        B, _, h_amap, w_amap = features.shape[:4]
        A, C = self.num_anchors, self.num_classes

        # forward pass for inference
        conf_scores = out[:, 0:5*A:5, :, :]             # [B, A, H', W']
        conf_scores = torch.sigmoid(conf_scores)

        offsets = out[:, 0:5*A, :, :].clone()           # [B, 5A, H', W']
        offsets = offsets.view(B, A, 5, h_amap, w_amap) # [B, A, 5, H', W']
        offsets = offsets[:, :, 1:5, :, :]              # [B, A, 4, H', W']
        offsets[:, :, :2, :, :] = torch.sigmoid(offsets[:, :, :2, :, :]) - 0.5

        cls_scores = out[:, 5*A:, :, :]                 # [B, C, H', W']

        # forward pass for training
        if pos_anc_idx is not None and neg_anc_idx is not None:
            conf_scores = self._extract_conf_scores(conf_scores, pos_anc_idx, neg_anc_idx)
            offsets = self._extract_offsets(offsets, pos_anc_idx)
            cls_scores = self._extract_cls_scores(cls_scores, pos_anc_idx)

        return conf_scores, offsets, cls_scores

    @staticmethod
    def _extract_conf_scores(conf_scores, pos_anc_idx, neg_anc_idx):
        """
        Extract confidence scores for positive and negative anchors.

        @Params:
        -------
        conf_scores (tensor):
            Tensor of shape [B, A, H', W'] given confidence scores for all anchors.
        pos_anc_idx (tensor):
            int64 tensor of shape [M] giving the indices of anchors marked as positive.
        neg_anc_idx (tensor):
            int64 tensor of shape [M] giving the indices of anchors marked as negative.

        @Returns:
        -------
        extracted_conf_score (tensor):
            Tensor of shape [2*M] giving the predicted confidence scores for positive anchors
            and negative anchors (in that order).
        """
        conf_scores = conf_scores.view(-1)
        pos_neg_idx = torch.cat([pos_anc_idx, neg_anc_idx])
        return conf_scores[pos_neg_idx]

    @staticmethod
    def _extract_offsets(offsets, pos_anc_idx):
        """
        Extract offsets for positive anchors.

        @Params:
        -------
        offsets (tensor):
            Tensor of shape [B, A, 4, H', W'] predicted transformations all all anchors.
        pos_anc_idx (tensor):
            int64 tensor of shape [M] giving the indices of anchors marked as positive.

        @Returns:
        -------
        extracted_offsets (tensor):
            Tensor of shape [M, 4] giving predicted transformation for positive anchors.
        """
        offsets = offsets.permute(0, 1, 3, 4, 2).reshape(-1, 4)
        return offsets[pos_anc_idx]

    def _extract_cls_scores(self, cls_scores, pos_anc_idx):
        """
        Extract class scores for positive anchors.

        @Params:
        ------
        cls_scores (tensor):
            Tensor of shape [B, C, H', W'] giving predicted classification scores for all anchors.
        pos_anc_idx (tensor):
            int64 tensor of shape [M] giving the indices of anchors marked as positive.

        @Returns:
        -------
        extracted_cls_scores (tensor):
            Tensor of shape [M, C) giving predicted classification scores for positive anchors.

        @Extra Info:
        Which and when to use `expand` and `repeat`?
        [Link](https://discuss.pytorch.org/t/torch-repeat-and-torch-expand-which-to-use/27969/2)
        """
        B, C, h_amap, w_amap = cls_scores.shape
        cls_scores = cls_scores.clone()                             # [B, C, H', W']
        cls_scores = cls_scores.permute(0, 2, 3, 1).contiguous()    # [B, H', W', C]
        cls_scores = cls_scores.unsqueeze(1).expand(B, self.num_anchors, h_amap, w_amap, C)  # [B, A, H', W', C]
        cls_scores = cls_scores.reshape(-1, C)
        return cls_scores[pos_anc_idx]


if __name__ == '__main__':
    from config import Configs
    model = FeatureExtractor(Configs, verbose=True)