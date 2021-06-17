import os
import sys


project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Configs:

    ###########
    # Datasets
    ###########
    # Root directory
    root_dir = os.path.join(project_dir, "data/VOCdevkit", "VOC2007")

    ##########
    # Anchors
    ##########
    # Number anchors per grid
    num_anchors = 9

    # Anchor shapes (on activation map)
    anchor_shapes = [
        [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [2, 3], [3, 2], [3, 5], [5, 3]
    ]

    #########
    # Models
    #########
    # Input shape
    input_height = 224
    input_width = 224
    input_shape = [3, input_height, input_width]

    # Feature Extractor
    strides = 32
