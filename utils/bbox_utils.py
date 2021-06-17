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
        Number of pixels in the width side of actication map.
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
    invalid_bbox_mask = (resized_bbox == -1)
    height_ratio = h_pixel * 1. / h_amap
    width_ratio = w_pixel * 1. / w_amap
    if mode == "p2a":
        # transfer from original image to activation map
        resized_bbox[..., [0, 2]] /= width_ratio
        resized_bbox[..., [1, 3]] /= height_ratio
    else:
        resized_bbox[..., [0, 2]] *= width_ratio
        resized_bbox[..., [1, 3]] *= height_ratio
    resized_bbox.mask_fill_(invalid_bbox_mask, -1)
    return resized_bbox

