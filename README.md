# SimDet
Pytorch version of Simple Object Detection for beginners to learn. 

*This repo is almost a solution of the [Assignment-5](https://web.eecs.umich.edu/~justincj/teaching/eecs498/FA2020/assignment5.html) from course [EECS 498/598 Deep Learning for Computer Vision](https://web.eecs.umich.edu/~justincj/teaching/eecs498/FA2020/schedule.html).*

## Content
- One Stage Object Detection: YOLO
- Two Stage Object Detection: Faster-RCNN

## Structure
```
.
├── config
├── data      # You need to manually download and extract Pascal VOC dataset into this folder 
│ └── VOCdevkit
│     └── VOC2007
│         ├── Annotations
│         ├── ImageSets
│         │ ├── Layout
│         │ ├── Main
│         │ └── Segmentation
│         ├── JPEGImages
│         ├── SegmentationClass
│         └── SegmentationObject
├── model
├── src
└── utils
```

