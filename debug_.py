import os
import cv2
import torch
from PIL import Image

from config import Configs
from src.dataset import get_pascal_voc_loader, VOCDataset, map_id_to_cls
from utils.visualize import visualize_detection
from utils.bbox_utils import coord_trans, generate_grids, get_anchor_shapes, generate_anchors


def visualize_voc_data():
    data_dir = Configs.root_dir
    voc = VOCDataset(data_dir, "train")
    ds_voc_valid = get_pascal_voc_loader(voc, batch_size=3)

    imgs, targets, h_list, w_list, img_id_list = next(iter(ds_voc_valid))
    for i in range(imgs.shape[0]):
        img = os.path.join(data_dir, "JPEGImages", img_id_list[i])
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bbox = targets[i]
        valid_mask = bbox[:, 0] != -1
        bbox = bbox[valid_mask]
        visualize_detection(img, bbox, map_id_to_cls=map_id_to_cls)


def visualize_grids():
    batch_size = 3
    data_dir = Configs.root_dir
    voc = VOCDataset(data_dir, "train")
    ds_voc_valid = get_pascal_voc_loader(voc, batch_size=batch_size)
    imgs, _, h_list, w_list, img_id_list = next(iter(ds_voc_valid))

    grids = generate_grids(3, device="cpu")
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
    voc = VOCDataset(data_dir, "train")
    ds_voc_valid = get_pascal_voc_loader(voc, batch_size=batch_size)
    imgs, _, h_list, w_list, img_id_list = next(iter(ds_voc_valid))

    anc_shapes = get_anchor_shapes(Configs)
    grid_centers = generate_grids(batch_size)
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


if __name__ == '__main__':

    # visualize_voc_data()

    visualize_grids()

    visualize_anchors()