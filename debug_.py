import os
import cv2
import random
import time
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

from config import Configs
from src.dataset import get_pascal_voc_loader, VOCDataset, map_id_to_cls
from src.solver import detection_solver
from modeling import SingleStageDetector
from utils.visualize import visualize_detection
from utils.pr_curve import precision_recall, pr_curve_vis, pr_auc
from utils.bbox_utils import (
    coord_trans, generate_grids, get_anchor_shapes, generate_anchors, generate_proposals, iou,
    reference_on_positive_anchors_yolo, nms_slow, nms_fast,
)


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
        visualize_detection(img, grids_xyxy[i].view(-1, 4), grids_cell[i].view(-1, 4), fix_color=True)


def visualize_anchors():
    # vis anchors at the center
    batch_size = 1
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

    # vis all anchors with shape [1, 1]
    print("Visualize anchors with shape [1, 1] ...")
    for i in range(imgs.shape[0]):
        img_path = os.path.join(data_dir, "JPEGImages", img_id_list[i])
        img = Image.open(img_path).convert("RGB")
        visualize_detection(img, anchors[i][2:3, ...].view(-1, 4))


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
                            fix_color=True)


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
    batch_size = 6
    data_dir = Configs.root_dir
    device = Configs.device
    imgs, targets, h_list, w_list, img_id_list = get_sample_data(batch_size)
    norm_targets = coord_trans(targets, h_list, w_list)

    anc_shapes = get_anchor_shapes(Configs)
    girds = generate_grids(batch_size, device=device)
    anchors = generate_anchors(anc_shapes, girds)
    iou_mat = iou(anchors, norm_targets)

    pos_anc_idx, neg_anc_idx, gt_conf_scores, gt_offsets, gt_cls_ids, pos_anc_coord, neg_anc_coord = \
        reference_on_positive_anchors_yolo(anchors, norm_targets, girds, iou_mat)

    num_anc_per_img = torch.prod(torch.tensor(anchors.shape[1:-1])).long()

    print("Number of anchors per image:", num_anc_per_img)
    print("Number positives:", pos_anc_idx.shape[0])
    print("Number of negatives:", neg_anc_idx.shape[0])

    # Positive Anchors
    for i in range(imgs.shape[0]):
        img_path = os.path.join(data_dir, "JPEGImages", img_id_list[i])
        img = Image.open(img_path).convert("RGB")
        anc_idx_in_img = (pos_anc_idx >= i * num_anc_per_img) & (pos_anc_idx < (i + 1) * num_anc_per_img)
        print(f"\n{i} - number positive anchors:", torch.sum(anc_idx_in_img))
        print("pos anchor index: ", pos_anc_idx[anc_idx_in_img])
        pos_anc_in_img = pos_anc_coord[anc_idx_in_img]
        pos_anc_in_img = coord_trans(pos_anc_in_img, h_list[i], w_list[i], mode="a2p")
        visualize_detection(img, targets[i, :, :4], pos_anc_in_img, map_id_to_cls, fix_color=True)

    # Negative Anchors
    for i in range(imgs.shape[0]):
        img_path = os.path.join(data_dir, "JPEGImages", img_id_list[i])
        img = Image.open(img_path).convert("RGB")
        anc_idx_in_img = (neg_anc_idx >= i * num_anc_per_img) & (neg_anc_idx < (i + 1) * num_anc_per_img)
        print(f"\n{i} - number negative anchors:", torch.sum(anc_idx_in_img))
        print("neg anchor index: ", neg_anc_idx[anc_idx_in_img])
        neg_anc_in_img = neg_anc_coord[anc_idx_in_img]
        neg_anc_in_img = coord_trans(neg_anc_in_img, h_list[i], w_list[i], mode="a2p")
        visualize_detection(img, targets[i, :, :4], neg_anc_in_img, map_id_to_cls, fix_color=True)


def overfit_small_data():
    num_sample = 10
    voc = VOCDataset(Configs.root_dir, "train")
    small_dataset = torch.utils.data.Subset(voc, torch.linspace(0, len(voc) - 1, steps=num_sample).long())
    small_train_loader = get_pascal_voc_loader(small_dataset, 8)

    for lr in [1e-2]:
        print("lr: lr")
        detector = SingleStageDetector(Configs)
        detection_solver(detector, small_train_loader, Configs)


def visualize_pr_curve():
    y_true = [1, 1, 0, 1, 0, 1, 1]
    y_pred = [1.00, 0.91, 0.91, 0.90, 0.87, 0.85, 0.00]

    prec, rec, thresh = precision_recall(y_pred, y_true)
    auc = pr_auc(prec, rec)

    #### 原始PR曲线
    fig = pr_curve_vis(prec, rec, smooth=False, title=f"Original PR-Curve")
    plt.show()

    #### 平滑后的结果
    fig = pr_curve_vis(prec, rec, smooth=True, title=f"Smoothed PR-Curve: AUC_{auc:.2f}")
    plt.show()


def test_nms(nms_func):
    print(f"\nTest the speed of `{nms_func.__name__}` against `torchvision.ops.nms")
    torch.manual_seed(0)
    random.seed(0)

    boxes = (100. * torch.rand(5000, 4)).round()
    boxes[:, 2] = boxes[:, 2] + boxes[:, 0] + 1.
    boxes[:, 3] = boxes[:, 3] + boxes[:, 1] + 1.
    scores = torch.randn(5000)

    names = ['your_cpu', 'torchvision_cpu', 'torchvision_cuda']
    iou_thresholds = [0.3, 0.5, 0.7]
    elapsed = dict(zip(names, [0.] * len(names)))
    intersects = dict(zip(names[1:], [0.] * (len(names) - 1)))

    for iou_threshold in iou_thresholds:
        tic = time.time()
        my_keep = nms_func(boxes, scores, iou_threshold)
        elapsed['your_cpu'] += time.time() - tic

        tic = time.time()
        tv_keep = torchvision.ops.nms(boxes, scores, iou_threshold)
        elapsed['torchvision_cpu'] += time.time() - tic
        intersect = len(set(tv_keep.tolist()).intersection(my_keep.tolist())) / len(tv_keep)
        intersects['torchvision_cpu'] += intersect

        tic = time.time()
        tv_cuda_keep = torchvision.ops.nms(boxes.cuda(), scores.cuda(), iou_threshold).to(my_keep.device)
        torch.cuda.synchronize()
        elapsed['torchvision_cuda'] += time.time() - tic
        intersect = len(set(tv_cuda_keep.tolist()).intersection(my_keep.tolist())) / len(tv_cuda_keep)
        intersects['torchvision_cuda'] += intersect

        print("NMS outputs:")
        print(my_keep[::100])
        print(tv_keep[::100])
        print(tv_cuda_keep[::100])
        print("\n\n")

    for key in intersects:
        intersects[key] /= len(iou_thresholds)

    # You should see < 1% difference
    print('Testing NMS:')
    print('Your        CPU  implementation: %fs' % elapsed['your_cpu'])
    print('torchvision CPU  implementation: %fs' % elapsed['torchvision_cpu'])
    print('torchvision CUDA implementation: %fs' % elapsed['torchvision_cuda'])
    print('Speedup CPU : %fx' % (elapsed['your_cpu'] / elapsed['torchvision_cpu']))
    print('Speedup CUDA: %fx' % (elapsed['your_cpu'] / elapsed['torchvision_cuda']))
    print('Difference CPU : ', 1. - intersects['torchvision_cpu'])  # in the order of 1e-3 or less
    print('Difference CUDA: ', 1. - intersects['torchvision_cuda'])  # in the order of 1e-3 or less


if __name__ == '__main__':
    # visualize_voc_data()

    # visualize_grids()

    # visualize_anchors()

    # visualize_yolo_xy_transform()
    # visualize_yolo_wh_transform()
    # visualize_faster_rcnn_xy_transform()
    # visualize_faster_rcnn_wh_transform()

    # visualize_pos_and_neg_anchors()

    # overfit_small_data()

    # visualize_pr_curve()

    # test_nms(nms_slow)
    test_nms(nms_fast)
