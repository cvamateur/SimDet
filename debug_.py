import os
import cv2
import torch
from PIL import Image

from config import Configs
from src.dataset import get_pascal_voc_loader, VOCDataset, map_id_to_cls
from utils.visualize import visualize_detection
from utils.bbox_utils import (coord_trans, generate_grids, get_anchor_shapes, generate_anchors, generate_proposals, iou,
                              _reference_on_positive_anchors_yolo)


def get_sample_data(n_samples=3):
    data_dir = Configs.root_dir
    device = Configs.device
    voc = VOCDataset(data_dir, "train")
    ds_voc_valid = get_pascal_voc_loader(voc, batch_size=n_samples)
    imgs, targets, h_list, w_list, img_id_list = next(iter(ds_voc_valid))
    targets = targets.to(device)
    h_list = h_list.to(device)
    w_list = w_list.to(device)
    return imgs, targets, h_list, w_list, img_id_list


def visualize_voc_data():
    data_dir = Configs.root_dir
    imgs, targets, h_list, w_list, img_id_list = get_sample_data(3)

    for i in range(imgs.shape[0]):
        img = Image.open(os.path.join(data_dir, "JPEGImages", img_id_list[i]))
        bbox = targets[i]
        valid_mask = bbox[:, 0] != -1
        bbox = bbox[valid_mask]
        visualize_detection(img, bbox, map_id_to_cls=map_id_to_cls)


def visualize_grids():
    data_dir = Configs.root_dir
    device = Configs.device
    imgs, _, h_list, w_list, img_id_list = get_sample_data(3)

    grids = generate_grids(3, device=device)
    grids_xyxy = torch.cat([grids, grids], dim=-1)
    grids_cell = grids_xyxy.clone()
    grids_cell[..., :2] -= 0.5
    grids_cell[..., 2:] += 0.5
    grids_xyxy = coord_trans(grids_xyxy, h_list, w_list, mode='a2p')
    grids_cell = coord_trans(grids_cell, h_list, w_list, mode='a2p')

    for i in range(imgs.shape[0]):
        img_path = os.path.join(data_dir, "JPEGImages", img_id_list[i])
        img = Image.open(img_path).convert("RGB")
        visualize_detection(img, grids_xyxy[i].view(-1, 4), grids_cell[i].view(-1, 4), fixed_color=True)


def visualize_anchors():
    # vis anchors at the center
    batch_size = 3
    data_dir = Configs.root_dir
    device = Configs.device
    imgs, _, h_list, w_list, img_id_list = get_sample_data(batch_size)

    anc_shapes = get_anchor_shapes(Configs)
    grid_centers = generate_grids(batch_size, device=device)
    anchors = generate_anchors(anc_shapes, grid_centers)
    anchors = coord_trans(anchors, h_list, w_list, mode="a2p")

    print("Visualize anchors at the center...")
    for i in range(imgs.shape[0]):
        img_path = os.path.join(data_dir, "JPEGImages", img_id_list[i])
        img = Image.open(img_path).convert("RGB")
        visualize_detection(img, anchors[i][:, 3:4, 3:4, :].view(-1, 4))

    # vis all anchors
    print("Visualize all anchors...")
    for i in range(imgs.shape[0]):
        img_path = os.path.join(data_dir, "JPEGImages", img_id_list[i])
        img = Image.open(img_path).convert("RGB")
        visualize_detection(img, anchors[i].view(-1, 4))


def visualize_proposals(offsets, method="YOLO"):
    batch_size = 3
    data_dir = Configs.root_dir
    device = Configs.device
    imgs, _, h_list, w_list, img_id_list = get_sample_data(batch_size)

    anc_shapes = get_anchor_shapes(Configs)
    grid_centers = generate_grids(batch_size, device=device)
    anchors = generate_anchors(anc_shapes, grid_centers)
    B, A, h_amap, w_amap = anchors.shape[:4]

    print(f"{method} ---- Visualize transform {offsets}...")
    offsets = offsets.view(1, 1, 1, 1, 4)
    offsets = offsets.repeat(B, A, h_amap, w_amap, 1)
    proposals = generate_proposals(anchors, offsets, method=method)
    anchors = coord_trans(anchors, h_list, w_list, mode="a2p")
    proposals = coord_trans(proposals, h_list, w_list, mode="a2p")
    for i in range(imgs.shape[0]):
        img_path = os.path.join(data_dir, "JPEGImages", img_id_list[i])
        img = Image.open(img_path).convert("RGB")
        visualize_detection(img,
                            anchors[i][0, 3:4, 3:4, :].view(-1, 4),
                            proposals[i][0, 3:4, 3:4, :].view(-1, 4),
                            fixed_color=True)


def visualize_yolo_xy_transform():
    offsets_xy = torch.tensor([0.5, 0.5, 0., 0.], device=Configs.device)
    visualize_proposals(offsets_xy, "YOLO")


def visualize_yolo_wh_transform():
    offsets_wh = torch.tensor([0., 0., 1., 1.], device=Configs.device)
    visualize_proposals(offsets_wh, "YOLO")


def visualize_faster_rcnn_xy_transform():
    offsets_xy = torch.tensor([1., 1., 0., 0.], device=Configs.device)
    visualize_proposals(offsets_xy, "FasterRCNN")


def visualize_faster_rcnn_wh_transform():
    offsets_wh = torch.tensor([0., 0., 1., 1.], device=Configs.device)
    visualize_proposals(offsets_wh, "FasterRCNN")


def visualize_pos_and_neg_anchors():
    batch_size = 8
    data_dir = Configs.root_dir
    device = Configs.device
    imgs, targets, h_list, w_list, img_id_list = get_sample_data(batch_size)
    norm_targets = coord_trans(targets, h_list, w_list)

    anc_shapes = get_anchor_shapes(Configs)
    girds = generate_grids(batch_size, device=device)
    anchors = generate_anchors(anc_shapes, girds)
    iou_mat = iou(anchors, norm_targets)

    pos_anc_idx, neg_anc_idx, gt_conf_scores, gt_offsets, gt_cls_ids, pos_anc_coord, neg_anc_coord = \
        _reference_on_positive_anchors_yolo(anchors, norm_targets, girds, iou_mat)

    num_anc_per_img = torch.prod(torch.tensor(anchors.shape[1:-1])).long()

    print("Number of anchors per image:", num_anc_per_img)

    # Positive Anchors
    for i in range(imgs.shape[0]):
        img_path = os.path.join(data_dir, "JPEGImages", img_id_list[i])
        img = Image.open(img_path).convert("RGB")
        anc_idx_in_img = (pos_anc_idx >= i * num_anc_per_img) & (pos_anc_idx < (i+1) * num_anc_per_img)
        print(f"\n{i} - number positive anchors:", torch.sum(anc_idx_in_img))
        print("pos anchor index: ", pos_anc_idx[anc_idx_in_img])
        pos_anc_in_img = pos_anc_coord[anc_idx_in_img]
        pos_anc_in_img = coord_trans(pos_anc_in_img, h_list[i], w_list[i], mode="a2p")
        visualize_detection(img, targets[i, :, :4], pos_anc_in_img, map_id_to_cls, fixed_color=True)

    # Negative Anchors
    for i in range(imgs.shape[0]):
        img_path = os.path.join(data_dir, "JPEGImages", img_id_list[i])
        img = Image.open(img_path).convert("RGB")
        anc_idx_in_img = (neg_anc_idx >= i * num_anc_per_img) & (neg_anc_idx < (i + 1) * num_anc_per_img)
        print(f"\n{i} - number negative anchors:", torch.sum(anc_idx_in_img))
        print("neg anchor index: ", neg_anc_idx[anc_idx_in_img])
        neg_anc_in_img = neg_anc_coord[anc_idx_in_img]
        neg_anc_in_img = coord_trans(neg_anc_in_img, h_list[i], w_list[i], mode="a2p")
        visualize_detection(img, targets[i, :, :4], neg_anc_in_img, map_id_to_cls, fixed_color=True)



if __name__ == '__main__':

    # visualize_voc_data()

    # visualize_grids()

    # visualize_anchors()

    # visualize_yolo_xy_transform()
    # visualize_yolo_wh_transform()
    # visualize_faster_rcnn_xy_transform()
    # visualize_faster_rcnn_wh_transform()

    visualize_pos_and_neg_anchors()
    pass