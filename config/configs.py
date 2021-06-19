import os
import sys
import torch

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Configs:

    ###########
    # Worldwide
    ###########
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ###########
    # Datasets
    ###########
    # Root directory
    root_dir = os.path.join(project_dir, "data/VOCdevkit", "VOC2007")
    num_classes = 20

    ##########
    # Anchors
    ##########
    # Number anchors per grid
    num_anchors = 9

    # Anchor shapes (on activation map)
    anchor_shapes = [
        [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [2, 3], [3, 2], [3, 5], [5, 3]
    ]

    # Thresholds for positive and negative anchors
    pos_thresh = 0.7
    neg_thresh = 0.3

    #########
    # Models
    #########
    # Input shape
    input_height = 224
    input_width = 224
    input_shape = [3, input_height, input_width]

    # Feature Extractor
    strides = 32
    features_dim = 1280

    #########
    # Loss
    #########
    # Weights of three losses
    w_conf = 1
    w_reg = 1
    w_cls = 1

    ########
    # Solver
    ########
    init_lr = 1e-3
    lr_decay = 1
    momentum = 0.9
    weight_decay = 5e-5
    epochs = 20

    ########
    # NMS
    ########

    # Threshold to eliminate redundant proposals
    nms_thresh = 0.7
