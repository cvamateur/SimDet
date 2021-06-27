import os

import torch

from config import Configs as cfg
from src.dataset import get_pascal_voc_loader, VOCDataset, map_id_to_cls
from src.solver import detection_solver
from modeling import SingleStageDetector


def main():

    print("Build VOC Dataset...")
    dataset_train = VOCDataset(cfg.root_dir, image_set="train")
    ds_train = get_pascal_voc_loader(dataset_train)
    print("VOC Dataset Built.\n")

    print("Build Detector...")
    detector = SingleStageDetector(cfg)
    print("Detector Built.\n")

    # Train the model
    detection_solver(detector, ds_train, cfg)

    # Save the model
    torch.save(detector.state_dict(), cfg.lst_ckpt)


if __name__ == '__main__':
    main()