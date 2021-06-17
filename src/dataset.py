import os
import torch
import xml.etree.ElementTree as ET

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, Callable, Tuple, List, Any

map_cls_to_id = {
    'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
    'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10,
    'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15,
    'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19
}
map_id_to_cls = {id: cls for cls, id in map_cls_to_id.items()}


class VOCDataset(Dataset):
    """Pascal VOC_0712 Detection Dataset.

    @Params
    -------
    root_dir (string):
        The root directory of VOC0712.
    image_set (string, optional):
        Specify which dataset split to use, `train`, `trainval`, `val`, `test`.
    transform (callable, optional):
        A function/transform that takes in an PIL image and returns a transformed version.
    """

    def __init__(self, root_dir: str, image_set: str = "val", transform: Optional[Callable] = None):
        assert os.path.isdir(root_dir), f"Invalid path: {root_dir}"
        assert image_set in ["train", "trainval", "val", "test"]
        assert transform is None or callable(transform)
        self.root_dir = root_dir
        self.image_set = image_set
        self.transform = transform

        split_dir = os.path.join(root_dir, "ImageSets/Main")
        split_f = os.path.join(split_dir, self.image_set + ".txt")
        with open(split_f, "r") as f:
            filenames = [line.strip() for line in f.readlines()]
        image_dir = os.path.join(root_dir, "JPEGImages")
        targets_dir = os.path.join(root_dir, "Annotations")
        self.images = [os.path.join(image_dir, filename + ".jpg") for filename in filenames]
        self.targets = [os.path.join(targets_dir, filename + ".xml") for filename in filenames]
        assert len(self.images) == len(self.targets)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Return a tuple of PIL-image[transformed] and annotation dict.

        @Params
        -------
        index (int): Index

        @Returns
        -------
        tuple: (img, target) where target is a dictionary of the xml tree.
        """
        img = Image.open(self.images[index]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        target = self._parse_voc_xml(ET.parse(self.targets[index]).getroot())
        return img, target["annotation"]

    def _parse_voc_xml(self, node: ET.Element) -> Dict[str, Any]:
        """Recursively parses xml contents to python dict.

        The tag which can appear multiple times is `object`.

        @Params
        -------
        node (ET.Element): a node object in xml tree.

        @Return
        -------
        dict: A dictionary contains all xml contents.
        """
        if len(node) == 0:
            return {node.tag: node.text}
        node_dict: Dict[str, Any] = {}
        for child in node:
            child_dict = self._parse_voc_xml(child)
            if child.tag != "object":
                node_dict[child.tag] = child_dict[child.tag]
            else:
                node_dict.setdefault(child.tag, []).append(child_dict[child.tag])
        return {node.tag: node_dict}


def collate_fn_voc(batch_list: List[Tuple[Any, Any]],
                   target_height=224,
                   target_width=224,
                   preprocess: Optional[Callable] = None):
    if preprocess is None:
        preprocess = transforms.Compose([
            transforms.Resize([target_height, target_width]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    batch_size = len(batch_list)
    max_num_box = max(len(batch_list[i][1]["object"]) for i in range(batch_size))
    img_batch = torch.zeros(batch_size, 3, target_height, target_width)
    bbox_batch = torch.full([batch_size, max_num_box, 5], fill_value=-1.)

    w_list = []
    h_list = []
    img_id_list = []
    for i in range(batch_size):
        img, ann = batch_list[i]
        w_list.append(img.size[0])
        h_list.append(img.size[1])
        img_id_list.append(ann["filename"])
        img_batch[i] = preprocess(img)
        for bbox_idx, one_bbox in enumerate(ann["object"]):
            bbox = one_bbox["bndbox"]
            xmin = float(bbox["xmin"])
            ymin = float(bbox["ymin"])
            xmax = float(bbox["xmax"])
            ymax = float(bbox["ymax"])
            cls_id = map_cls_to_id[one_bbox["name"]]
            bbox_batch[i][bbox_idx] = torch.Tensor([xmin, ymin, xmax, ymax, cls_id])

    h_batch = torch.tensor(h_list)
    w_batch = torch.tensor(w_list)
    return img_batch, bbox_batch, h_batch, w_batch, img_id_list


class DataAug(object):
    pass


def get_pascal_voc_loader(dataset, batch_size=8, shuffle=False, num_workers=0):
    """
    Data loader for Pascal VOC 2007.
    https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    """
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      pin_memory=True,
                      num_workers=num_workers,
                      collate_fn=collate_fn_voc)
