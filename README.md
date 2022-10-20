# Object Detection Pipeline

Object detection is a computer technology related to computer vision and image processing that deals with detecting 
instances of semantic objects of a certain class in digital images and videos. 
Well-researched domains of object detection include face detection and pedestrian detection.

## Steps to train object detection using pretrained models:
1. Dataset preparation 
2. Augmentation 
3. Select Architecture like YOLO, SSD and RCNN
4. Train
5. Evaluate


## Steps to train YOLOv5 with Person class: 
1. In order to clone repository follow steps mention in this document -> https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
2. Download Person Dataset  
3. Download Preprocessing Scripts  
4. Create and modify person_class.yaml with the help of  data/coco128.yaml 
5. Modify “create_dataset_for_unique_classes.py” and choose classes that need to be trained and Run.
6. Modify “xml_to_csv.py” and Run.
7. python train.py --img 640 --batch 8 --epochs 10 --data person_class.yaml  --weights yolov5s.pt
8. python detect.py --weights "runs/train/exp/weights/best.pt" --source "V_Jan_2000" --data "data/person_class.yaml"
### In order to load model with opencv dnn: 
1. python export.py --weights yolov5s.pt --include onnx
2. Use utility -> YOLOv5/detect_object_with_yolov5_onnx.py

## Steps to train SSD-VGG caffe model:
Please refer this link -> https://github.com/prabodh78/SSD_Caffe_Model_Training