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

    resized_bbox = bbox.clone()
    resized_bbox = resized_bbox.view(bbox.shape[0], -1, bbox.shape[-1])
    invalid_bbox_mask = (resized_bbox == -1)
    height_ratio = h_pixel * 1. / h_amap
    width_ratio = w_pixel * 1. / w_amap
    height_ratio = height_ratio.to(resized_bbox.device)
    width_ratio = width_ratio.to(resized_bbox.device)

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
    return torch.tensor(cfg.anchor_shapes)


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

