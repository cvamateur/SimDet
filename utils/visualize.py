import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import colorsys
import torch
from PIL.Image import Image


def get_cmap(n, name='prism'):
    """
    Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGBA color; the keyword argument name must be a standard mpl colormap name.
    [ColorMap](https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.Colormap.html#matplotlib.colors.Colormap)
    [ColorMap-Names](https://matplotlib.org/stable/gallery/color/colormap_reference.html)

    @Params
    -------
    n (int):
        Maps colors into a range of distinct int values [0, n-1]
    name (string):
        colorMap names, refer to the link `[ColorMap-Names]` for more details.

    @Returns
    -------
        A colorMap instance, which is a callable function that takes an int number and returns a tuple
        of RGBA color.


    @Usage
    ------
    cmap = get_cmap(len(data))
    for i, (X, Y) in enumerate(data):
        scatter(X, Y, c=cmap(i))
    """
    return plt.cm.get_cmap(name, n)


def _draw_bbox_on_image(image, bbox, color, map_id_to_cls=None, fix_color=False):
    """
    Draw one bbox on image.
    """
    cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
    if len(bbox) > 4:
        cls_id = int(bbox[4])
        cls_name = ""
        cls_conf = ""
        if map_id_to_cls is not None:
            cls_name = map_id_to_cls[cls_id]
        if len(bbox) > 5:
            cls_conf = f", {bbox[5]: .2f}"
        tag = cls_name + cls_conf
        if not fix_color: color = change_lightness_color(color, 2.)
        cv2.putText(image, tag, (int(bbox[0]), int(bbox[1]) + 15), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 1)


def change_lightness_color(color, factor: float):
    """
    Darken or lighten a RGB color according to factor.

    @Params:
    -------
    color (Tuple[R,G,B]):
        a tuple of RGB color.
    factor (float):
        If factor < 1, then the lightened color is returned, otherwise
        the darkened color is returned.
    """
    h, l, s = colorsys.rgb_to_hls(*color)
    return colorsys.hls_to_rgb(h, 1 - factor * (1 - l), s)


def visualize_detection(image, bbox=None, pred=None, map_id_to_cls=None, fix_color=False):
    """
    Visualize bbox on the original image.

    @Params
    -------
    image (PIL Image):
        Original image to visualize.
    map_id_to_cls (dict):
        Mapping from class id to class name.
    bbox (tensor)
        GT bbox, tensor of shape [N, 5], where 5 indicates
            (x_tl, y_tl, x_br, y_br, cls_id)
    pred (tensor)
        Predicted bbox, tensor of shape [N, 6], where 6 indicates
            (x_tl, y_tl, x_br, y_br, cls_id, cls_conf_score)
    """
    image = np.array(image).astype(np.uint8)
    n = 0
    if bbox is not None:
        n = max(n, bbox.shape[0])
    if pred is not None:
        n = max(n, pred.shape[0])
    cmap = get_cmap(n)

    if bbox is not None:
        # draw GT bbox
        for i in range(bbox.shape[0]):
            one_bbox = bbox[i]

            if fix_color:
                color = (255, 0, 0)
            else:
                color = tuple(map(lambda x: int(x * 255.), cmap(i)[:3]))
            _draw_bbox_on_image(image, one_bbox, color, map_id_to_cls)

    if pred is not None:
        # draw predicted bbox
        for i in range(pred.shape[0]):
            one_bbox = pred[i]
            if fix_color:
                color = (0, 255, 0)
            else:
                color = tuple(map(lambda x: int(x * 255.), cmap(i)[:3]))
                color = change_lightness_color(color, 0.5)
            _draw_bbox_on_image(image, one_bbox, color, map_id_to_cls)

    plt.imshow(image)
    plt.axis("off")
    plt.show()
