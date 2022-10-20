from xml.dom import minidom
import os
import glob

import cv2
font = cv2.FONT_HERSHEY_PLAIN


def convert_coordinates(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_xml2yolo(lut, dir_path, txt_output_dir):
    print(os.path.exists(dir_path))
    for fname in glob.glob(dir_path + "/*.xml"):
        xmldoc = minidom.parse(fname)
        fn_, ext_ = os.path.splitext(os.path.basename(fname))
        fname_out = os.path.join(txt_output_dir, (fn_ + '.txt'))
        with open(fname_out, "w") as f:

            itemlist = xmldoc.getElementsByTagName('object')
            size = xmldoc.getElementsByTagName('size')[0]
            width = int((size.getElementsByTagName('width')[0]).firstChild.data)
            height = int((size.getElementsByTagName('height')[0]).firstChild.data)

            for item in itemlist:
                # get class label
                classid = (item.getElementsByTagName('name')[0]).firstChild.data
                if classid in lut:
                    label_str = str(lut[classid])

                    # get bbox coordinates
                    xmin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmin')[0]).firstChild.data
                    ymin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymin')[0]).firstChild.data
                    xmax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmax')[0]).firstChild.data
                    ymax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymax')[0]).firstChild.data
                    b = (float(xmin), float(xmax), float(ymin), float(ymax))
                    bb = convert_coordinates((width, height), b)
                    # print(bb)
                    f.write(label_str + " " + " ".join([("%.6f" % a) for a in bb]) + '\n')

                else:
                    label_str = "-1"
                    print("warning: label '%s' not in look-up table" % classid)

    print("wrote %s" % fname_out)


def test(file_=None, img_name=None):
    if file_ is None:
        file_ = '/home/prabodh/personal_space/yolov5/data/coco128/labels/train/6257.txt'
    if img_name is None:
        img_name = '/home/prabodh/personal_space/yolov5/data/coco128/images/train/6257.jpg'
    # img_name = os.path.join(dest_label_p, img_name)
    # os.path.exists(os.path.join(dest_label_p, img_name))

    img = cv2.imread(img_name)
    labels = []
    with open(file_, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            line = [float(value) for value in line.split()]
            labels.append(line)

    h, w, c = img.shape
    for label in labels:
        x_c = label[1] * w
        y_c = label[2] * h
        x_w = label[3] * w
        y_h = label[4] * h

        x1 = int(x_c - (x_w / 2))
        y1 = int(y_c - (y_h / 2))
        x3 = int(x_c + (x_w / 2))
        y3 = int(y_c + (y_h / 2))
        print(label[0])
        cv2.putText(img, str(label[0]), (x1, y1), font, 0.5, (0, 0, 0), 1)
        cv2.rectangle(img, (x1, y1), (x3, y3), (0, 255, 0), 1)
    cv2.imshow('Img', img)
    cv2.waitKey(0)


# ['bottle', 'cellphone', 'face', 'glass', 'hand', 'mug', 'person']
classes = {'person': 0}
dir_path = '/home/prabodh/workspace/Person_Detector/person_face_dataset_v1/person_voco_client_v1/images/train2'
txt_output_dir = '/home/prabodh/workspace/Person_Detector/person_face_dataset_v1/person_voco_client_v1/labels/train2'
convert_xml2yolo(classes, dir_path, txt_output_dir)
images = glob.glob('{}/*.jpg'.format(dir_path))
test(images[0].replace('images', 'labels').replace('jpg', 'txt'), images[0])