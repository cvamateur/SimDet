import torch
import torch.nn.functional as F


def loss_conf_scores_regression(conf_score, gt_conf_scores):
    """
    Use sum-squared error as in YOLO.

    @Params:
    -------
    conf_score (tensor):
        Predicted confidence score of shape [2M, 1]
    gt_conf_scores (tensor):
        GT conf scores of shape [M]

    @Returns:
    -------
    loss_conf_score (scalar tensor):
       Loss of confidence score.
    """
    # The target conf score are always 1's for positives and 0's for negatives
    gt_conf_scores = torch.cat([torch.ones_like(gt_conf_scores), torch.zeros_like(gt_conf_scores)])
    return torch.mean((conf_score - gt_conf_scores) ** 2)


def loss_bbox_regression(offsets, gt_offsets):
    """
    Use sum-squared error as in YOLO.

    @Params:
    -------
    offsets (tensor):
        Predicted bbox offsets of shape [M, 4].
    gt_offsets (tensor):
        GT bbox offsets of shape [M, 4].

    @Returns:
    -------
    loss_bbox_reg (scalar tensor):
        Loss of bbox regression.
    """
    return torch.mean(torch.sum((offsets - gt_offsets)**2, dim=1))


def loss_cls_ce(cls_id, gt_cls_id, batch_size, num_anc_per_img, pos_anc_idx):
    """
    Cross-entropy loss for objects classification.

    @Params:
    ------
    cls_id (tensor):
        Predicted class scores of shape [M, C].
    gt_cls_id (tensor):
        GT class scores of shape [M].
    num_anc_per_img (int):
        Number of anchors on each image, equals to A*H'*W'.
    pos_anc_idx (tensor):
        int64 tensor of shape [M] giving the indices of anchors marked as positive.

    @Returns:
    -------
    obj_cls_loss (scalar tensor):
        Loss of object classification.
    """
    all_loss = F.cross_entropy(cls_id, gt_cls_id, reduction="none")
    obj_cls_loss = 0.
    for i in range(batch_size):
        anc_idx_in_img = (pos_anc_idx >= i * num_anc_per_img) & (pos_anc_idx < (i+1) * num_anc_per_img)
        obj_cls_loss += all_loss[anc_idx_in_img].sum() / anc_idx_in_img.sum()
    obj_cls_loss /= batch_size
    return obj_cls_loss
