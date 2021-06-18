import torch


def coord_trans(bbox, h_pixel, w_pixel, h_amap=7, w_amap=7, mode="p2a"):
    """
    Coordinate transformation function. It converts the box coordinate from
    the image coordinate system to the activation map coordinate system and
    vice versa.

    @Params
    -------
    bbox (tensor):
        Tensor of shape [B, N, 4], where 4 indicates (x_tl, y_tl, x_br, y_br).
    h_pixel (list or tensor):
        Tensor of shape [B], containing heights of each image.
    w_pixel (list or tensor):
        Tensor of shape [B], containing width of each image.
    h_amap (int):
        Number of pixels in the height side of activation map.
    w_amap (int):
        Number of pixels in the width side of activation map.
    mode (string):
        Indicate which mode the transformation is done. Either `p2a` or `a2p`,
        where `p2a` means transfer from original image to activation map;
              `a2p` means transfer from activation map to original image.

    @Returns
    -------
    resized_bbox (tensor):
        A resized bbox tensor of the same shape as input bbox, [B, N, 4]
    """
    assert mode in ["p2a", "a2p"], "Invalid transformation mode!"
    assert bbox.shape[-1] >= 4, "Last dim must be larger or equal then 4!"

    # handle corner case
    if bbox.shape[0] == 0:
        return bbox

    if isinstance(h_pixel, list):
        h_pixel = torch.tensor(h_pixel, dtype=bbox.dtype, device=bbox.device)
    if isinstance(w_pixel, list):
        w_pixel = torch.tensor(w_pixel, dtype=bbox.dtype, device=bbox.device)

    resized_bbox = bbox.clone()
    resized_bbox = resized_bbox.view(bbox.shape[0], -1, bbox.shape[-1])
    invalid_bbox_mask = (resized_bbox == -1)
    height_ratio = h_pixel * 1. / h_amap
    width_ratio = w_pixel * 1. / w_amap
    # height_ratio = height_ratio.to(resized_bbox.device)
    # width_ratio = width_ratio.to(resized_bbox.device)

    if mode == "p2a":
        # transfer from original image to activation map
        resized_bbox[..., [0, 2]] /= width_ratio.view(-1, 1, 1)
        resized_bbox[..., [1, 3]] /= height_ratio.view(-1, 1, 1)
    else:
        resized_bbox[..., [0, 2]] *= width_ratio.view(-1, 1, 1)
        resized_bbox[..., [1, 3]] *= height_ratio.view(-1, 1, 1)

    resized_bbox.masked_fill_(invalid_bbox_mask, -1)
    resized_bbox.resize_as_(bbox)
    return resized_bbox


def get_anchor_shapes(cfg):
    """Return anchor shapes.

    @Params:
    -------
    cfg (config.Config):
        An instance of Config, in which cfg.anchor_shapes is a list of anchor shapes.
    """
    assert hasattr(cfg, "anchor_shapes"), "cfg has no attribute named `anchor_shapes`!"
    return torch.tensor(cfg.anchor_shapes).to(cfg.device)


def generate_grids(batch_size, h_amap=7, w_amap=7, device="cuda"):
    """
    Generate grid centers.

    @Parmas:
    -------
    batch_size (int):
        Batches of grids to return.
    h_amap (int):
        Height of the activation map (number of grids in vertical dimension).
    w_amap (int):
        Width of the activation map (number of grids in horizontal dimension).
    device (string):
        Device of the returned tensor.

    @Returns:
    -------
    grids (tensor):
        A float32 tensor of shape [B, h_amap, w_amap, 2] giving (cx, cy) coordinates
        of the centers of each feature for a feature map of shape (B, D, h_amap, w_amap)
    """
    h_range = torch.arange(h_amap, dtype=torch.float32, device=device) + 0.5
    w_range = torch.arange(w_amap, dtype=torch.float32, device=device) + 0.5

    grid_ys = h_range.view(-1, 1).repeat(1, w_amap)
    grid_xs = w_range.view(1, -1).repeat(h_amap, 1)
    grid_centers = torch.stack([grid_xs, grid_ys], dim=-1)
    grid_centers = grid_centers.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    return grid_centers


def generate_anchors(anchor_shapes, grid_centers):
    """
    Generate all anchors.

    @Params:
    -------
    anchor_shapes (tensor):
        Tensor of shape [A, 2] giving the shapes of anchor boxes to consider at each
        point in the grid. anchor_shapes[a] = (w, h) indicate width and height
        of the a-th anchors.
    grid_centers (tensor):
        Tensor of shape [B, H', W', 2] giving the (x, y) coordinates of the center of
        each feature from the backbone feature-map. This is the tensor returned by
        `generate_grids()`

    @Returns:
    -------
    anchors (tensor):
        Tensor of shape [B, A, H', W', 4] giving the positions of all anchor boxes for
        the entire image. anchors[b, a, h, w] is an anchor box centered at  grid_centers[b, h, w],
        whose shape is given by anchor_shapes[a]. The anchor boxes are parameterized as
        (x_tl, y_tl, x_br, y_br).
    """
    B, h_amap, w_amap = grid_centers.shape[:3]
    A = anchor_shapes.shape[0]
    device = grid_centers.device
    anchor_shapes = anchor_shapes.view(1, A, 1, 1, 2).to(device)
    grid_centers = grid_centers.unsqueeze(1)

    anchors = torch.zeros([B, A, h_amap, w_amap, 4], dtype=torch.float32, device=device)
    anchors[..., :2] = grid_centers - anchor_shapes / 2.
    anchors[..., 2:] = grid_centers + anchor_shapes / 2.
    return anchors


def generate_proposals(anchors, offsets, method="YOLO"):
    """
    Generate all proposals from anchors given offsets.

    @Params:
    -------
    anchors (tensor):
        Tensor of shape [B, A, H', W', 4] returned by `generate_anchors()`. Anchors are
        parameterized by the coordinates (x_tl, y_tl, x_br, y_br).
    offsets (tensor):
        Transformations of the same shape as anchors, [B, A, H', W', 4], that will be
        used to convert anchor boxes into region proposals. The formula the transformation
        from offsets[b, a, h, w] = (tx, ty, th, tw) to anchors[b, a, h, w] = (x_tl, y_tl, x_br, y_bt)
        will be difference according to `method`:
            method == "YOLO":               #  Assume values in range
                cx_p = cx_a + tx            # `tx` in range [-0.5, 0.5]
                cy_p = cy_a + ty            # `ty` in range [-0.5, 0.5]
                w_p = w_a * e^tw            # `tw` in range [-inf, +inf]
                h_p = h_a * e^th            # `th` in range [-inf, +inf]

            method == "FasterRCNN":         #  Assume values in range
                cx_p = cx_a + tx * w_a      # `tx` in range [-inf, +inf]
                cy_p = cy_a + ty * h_a      # `ty` in range [-inf, +inf]
                w_p = w_a * e^tw            # `tw` in range [-inf, +inf]
                h_p = h_a * e^th            # `th` in range [-inf, +inf]
    method (string):
        Indicate which transformation to apply, either "YOLO" or "FasterRCNN"

    @Returns:
    -------
    proposals (tensor):
        Region proposals, tensor of shape [B, A, H', W', 4], represented by the coordinates of
        (x_tl, y_tl, x_br, y_br).
    """
    assert method in ["YOLO", "FasterRCNN"], "Invalid method, either `YOLO` or `FasterRCNN`!"

    # 1. Convert anchors from (x_tl, y_tl, x_br, y_br) to (cx, cy, w, h)
    anchors_xywh = torch.zeros_like(anchors)
    anchors_xywh[..., :2] = anchors[..., 2:] / 2. + anchors[..., :2] / 2.
    anchors_xywh[..., 2:] = anchors[..., 2:] - anchors[..., :2]

    # 2. Apply transformation
    proposals_xywh = torch.zeros_like(anchors)
    if method == "YOLO":
        proposals_xywh[..., :2] = anchors_xywh[..., :2] + offsets[..., :2]
        proposals_xywh[..., 2:] = anchors_xywh[..., 2:] * torch.exp(offsets[..., 2:])
    else:
        proposals_xywh[..., :2] = anchors_xywh[..., :2] + anchors_xywh[..., 2:] * offsets[..., :2]
        proposals_xywh[..., 2:] = anchors_xywh[..., 2:] * torch.exp(offsets[..., 2:])

    # 3. Convert proposals from (cx, cy, w, h) back to (x_tl, y_tl, x_br, y_br)
    proposals = torch.zeros_like(proposals_xywh)
    proposals[..., :2] = proposals_xywh[..., :2] - proposals_xywh[..., 2:] / 2.
    proposals[..., 2:] = proposals_xywh[..., :2] + proposals_xywh[..., 2:] / 2.
    return proposals


def iou(proposals, gt_bboxes):
    """
    Compute Intersection over Union between sets of bounding boxes.

    @Params:
    -------
    proposals (tensor):
        Proposals of shape [B, A, H', W', 4], where 4 indicates (x_tl, y_tl, x_br, y_br)
    gt_bboxes (tensor):
        Ground truth boxes, from the DataLoader, of shape [B, N, 5], where 5 indicates
        (x_tl, y_tl, x_br, y_br, cls_id). N is the max number of bboxes within this batch,
        for images[i] which has fewer bboxes then N, then gt_bboxes[i] will be padded
        with extra rows of -1.

    @Returns:
    -------
    iou_mat (tensor):
        IoU matrix of shape [B, A*H'*W', N] where iou_mat[b, i, n] gives the IoU between
        one element of proposals[b] with gt_bboxes[b, n]
    """
    B, A, h_amap, w_amap = proposals.shape[:4]
    N = gt_bboxes.shape[1]
    proposals = proposals.view(B, -1, 1, 4)           # [B, A*H'*W', 1, 4]
    gt_bboxes = gt_bboxes[..., :4].view(B, 1, -1, 4)  # [B, 1, N, 4]

    # Area of proposals, shape [B, A*H'*W', 1]
    proposals_wh = proposals[..., 2:] - proposals[..., :2]
    proposals_area = proposals_wh[..., 0] * proposals_wh[..., 1]

    # Area of gt_bboxes, shape [B, 1, N]
    gt_bboxes_wh = gt_bboxes[..., 2:] - gt_bboxes[..., :2]
    gt_bboxes_area = gt_bboxes_wh[..., 0] * gt_bboxes_wh[..., 1]

    # Area of Intersection, shape [B, A*H'*W', N]
    intersect_xy_tl = torch.maximum(proposals[..., :2], gt_bboxes[..., :2])
    insersect_xy_br = torch.minimum(proposals[..., 2:], gt_bboxes[..., 2:])
    intersect_wh = (insersect_xy_br - intersect_xy_tl).clamp(min=0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]  # [B, A*H'*W', N]

    # Area of Union
    union = proposals_area + gt_bboxes_area - intersect_area

    # IoU
    iou_mat = intersect_area / union
    return iou_mat


def _reference_on_positive_anchors_yolo(anchors, gt_bboxes, grids, iou_mat, neg_thresh=0.3):
    """
    Determine the positive and negative anchors for model training.

    For YOLO, a grid cell is responsible for predicting GT box if the center of the box falls
    into that grid.
    Implement Details:
        First compute manhattan distance between grid centers and gt_bboxes. This gives us a
        matrix of shape [B, H'*W', N], then perform `torch.min(dim=1)[1]` on it gives us the
        indexes indicating positive grids responsible for GT boxes.
        Second, among all the anchors associated with the positive grids, the anchor with the
        largest IoU with the GT box is responsible to predict the GT box.
    NOTE: One anchor might match multiple GT boxes.

    Negative labels are assigned to those anchors which has IoU lower than `neg_thresh`.
    Anchors that are neither positive or negative are neutral, which do not contribute to
    the training objective.

    Main steps include:
    1) Decide positive and negative anchors based on iou_mat;
    2) Compute GT conf_score/GT offsets/Gt cls_id on the positive anchors
    3) Compute GT conf_score for negative anchors

    @Params:
    -------
    anchors (tensor):
        Anchors of shape [B, A, H', W, 4], returned by `generate_anchors()`.
    gt_bboxes (tensor):
        GT bbox of shape [B, N, 5] returned by DataLoader, where N is PADDED GT Boxes, 5
        indicate (x_tl, y_tl, x_br, y_br, cls_id)
    grids (tensor):
        Grid centers of shape [B, H', W', 2] where 2 indicate (x, y) coord.
    iou_mat (tensor):
        IoU matrix of shape [B, A*H'*W', N] where A is number of anchors each grid contains.
    neg_thresh (float):
        Threshold for specify negative anchors.

    @Returns:
    -------
    pos_anc_idx (tensor):
        Index on positive anchors, of shape [M], where M is the number of positive anchors.
    neg_anc_idx (tensor):
        Index on negative anchors, of shape [M].
    gt_conf_scores (tensor):
        GT IoU scores on positive anchors, of shape [M].
    gt_offsets (tensor):
        GT offsets on positive anchors, of shape [M, 4], denoted as (\hat{tx}, \hat{ty},
        \hat{tw}, \hat{th}). Refer to `generate_proposals()` for the transformation.
    gt_cls_ids (tensor):
        GT class ids on positive anchors, of shape [M].
    pos_anc_coord (tensor):
        Coordinates on positive anchors (mainly for visualization purpose).
    neg_anc_coord (tensor):
        Coordinates on negative anchors (mainly for visualization purpose).
    """
    B, A, h_amap, w_amap = anchors.shape[:4]
    N = gt_bboxes.shape[1]

    bbox_mask = (gt_bboxes[:, :, 0] != -1)   # [B, N]
    bbox_centers = (gt_bboxes[..., 2:4] - gt_bboxes[..., :2]) / 2. + gt_bboxes[..., :2]  # [B, N, 2]

    ######### Positive #########
    # L1 distances between girds centers and gt bboxes centers, of shape [B, H'*W', N]
    mah_dist = torch.sum(torch.abs(grids.view(B, -1, 1, 2) - bbox_centers.unsqueeze(1)), dim=-1)
    # Get the minimum dist for each grid
    min_mah_dist = torch.min(mah_dist, dim=1, keepdim=True)[0]  # [B, 1, N]
    # positive grid
    grid_mask = (mah_dist == min_mah_dist).unsqueeze(1)         # [B, 1, H'*W', N]

    # Get the maximum IoU among all `A` anchors for each grid
    reshaped_iou_mat = iou_mat.view(B, A, -1, N)                # [B, A, H'*W', N]
    anc_with_max_iou = torch.max(reshaped_iou_mat, dim=1, keepdim=True)[0]  # [B, 1, H'*W', N]
    anc_mask = (reshaped_iou_mat == anc_with_max_iou)           # [B, A, H'*W', N]

    # Get positive anchors, those on minimum dist grids as well as max IoU
    pos_anc_mask = (grid_mask & anc_mask).view(B, -1, N)        # [B, A*H'*W', N]
    pos_anc_mask = pos_anc_mask & bbox_mask.view(B, 1, N)       # [B, A*H'*W', N]

    # Get positive anchor indexes. NOTE: one anchor could match multiple GT boxes
    pos_anc_idx = torch.nonzero(pos_anc_mask.view(-1)).squeeze(-1)  # [M]

    # GT conf scores
    gt_conf_scores = iou_mat.view(-1)[pos_anc_idx]                  # [M]

    # GT class ids
    # First expand GT bboxes to the same shape as anchors
    gt_bboxes = gt_bboxes.view(B, 1, N, 5).repeat(1, A*h_amap*w_amap, 1, 1).view(-1, 5)
    gt_cls_ids = gt_bboxes[:, 4][pos_anc_idx].long()
    gt_bboxes = gt_bboxes[:, :4][pos_anc_idx]

    ##### IMPORTANT #####
    pos_anc_idx = (pos_anc_idx / float(N)).long()
    #####################

    # Get positive anchor coord
    pos_anc_coord = anchors.view(-1, 4)[pos_anc_idx]

    # GT offsets
    gt_offsets_xy = (gt_bboxes[:, :2] + gt_bboxes[:, 2:] - pos_anc_coord[:, :2] - pos_anc_coord[:, 2:]) / 2.
    gt_offsets_wh = torch.log((gt_bboxes[:, 2:] - gt_bboxes[:, :2]) / (pos_anc_coord[:, 2:] - pos_anc_coord[:, :2]))
    gt_offsets = torch.cat([gt_offsets_xy, gt_offsets_wh], dim=-1)

    ########## Negative ##########
    neg_anc_mask = iou_mat.view(B, -1) < neg_thresh    # [B, A*H'*W'*N]
    neg_anc_idx = torch.nonzero(neg_anc_mask.view(-1)).squeeze(-1)
    neg_anc_idx = (neg_anc_idx / float(N)).long()
    rand_pick = torch.randint(neg_anc_idx.shape[0], size=[pos_anc_idx.shape[0]])
    neg_anc_idx = neg_anc_idx[rand_pick]
    neg_anc_coord = anchors.view(-1, 4)[neg_anc_idx]

    return pos_anc_idx, neg_anc_idx, gt_conf_scores, gt_offsets, gt_cls_ids, pos_anc_coord, neg_anc_coord
