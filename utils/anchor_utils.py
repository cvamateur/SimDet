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

