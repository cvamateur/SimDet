import torch


def get_anchor_shapes(cfg):
    """Return anchor shapes.

    @Params:
    -------
    cfg (config.Config):
        An instance of Config, in which cfg.anchor_shapes is a list of anchor shapes.
    """
    assert hasattr(cfg, "anchor_shapes"), "No attribute `anchor_shapes` in cfg!"
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
    grid_centers (tensor):
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

