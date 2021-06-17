import os

from config import Configs
from src.dataset import get_pascal_voc_loader, VOCDataset, map_id_to_cls
from utils.visualize import visualize_detection


if __name__ == '__main__':
    import os
    # Visualize VOC0712
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
        visualize_detection(img, map_id_to_cls, bbox)