import cv2
import numpy as np
import time
import glob
import os
import torch
import torchvision
import logging as LOGGER
import random
import csv

# classes = ['person', 'face', 'cellphone', 'hand']
dir_name = os.path.dirname(os.path.abspath(__file__))
names = os.path.join(dir_name, 'ms_coco_dataset.txt')

with open(names, "r") as f:
    classes = []
    for line in f.readlines():
        if line.strip() != '':
            classes.append(line.strip())
CLASSES = ['person']
color_plans = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
# print(len(classes, len(color_plans)))

colors = np.random.uniform(0, 255, size=(len(classes), 3))
font = cv2.FONT_HERSHEY_PLAIN
path = os.path.dirname(os.path.abspath(__file__))


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[:, [0, 2]] -= pad[0]  # x padding
    boxes[:, [1, 3]] -= pad[1]  # y padding
    boxes[:, :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_area(box):
    # box = xyxy(4,n)
    return (box[2] - box[0]) * (box[3] - box[1])


def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter + eps)


def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    prediction = torch.tensor(prediction)
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
        else:
            x = x[x[:, 4].argsort(descending=True)]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def yolov5_prediction(image_path, model=None, threshold=0.4, nms_threshold=0.45, debug=False):
    final_boxes = {i: [] for i in range(0, len(CLASSES) + 1)}
    final_confidence = {i: [] for i in range(0, len(CLASSES) + 1)}
    all_labels = []

    if not model:
        # model = cv2.dnn.readNetFromONNX(os.path.join(dir_name, 'deep_models/yolov5s.onnx'))
        model = cv2.dnn.readNetFromONNX('/home/prabodh/workspace/Person_Detector/models/yolov5/yolov5s-person_voco_client_v1_epochs-4.onnx')
    img = cv2.imread(image_path)
    image_write = img.copy()
    height, width = img.shape[:2]
    if True:
        im = letterbox(img, new_shape=[640, 640], stride=32, auto=False)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        # im = im.float()
        im = np.array(im, dtype='float32')
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

    model.setInput(im)
    y = model.forward()
    if isinstance(y, (list, tuple)):
        y_pred = y[0] if len(y) == 1 else [x for x in y]
    else:
        y_pred = y

    # Applying NMS:
    conf_thres, iou_thres, agnostic_nms = (threshold, nms_threshold, False)
    y_pred = non_max_suppression(y_pred, conf_thres, iou_thres, None, agnostic_nms, max_det=1000)
    gn = torch.tensor(img.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for i, det in enumerate(y_pred):  # per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                det = np.copy(det)
                if float(conf) > threshold and classes[int(cls)] in CLASSES:
                    # normalized_rects = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                    rects = [int(i) for i in xyxy]
                    x1, y1, x2, y2 = rects
                    rects = [x1, y1, x2 - x1, y2 - y1]
                    startX, startY, endX, endY = rects
                    if startX < 0:
                        startX = 0
                    if endX > img.shape[1] - 1:
                        endX = img.shape[1] - 1

                    if startY < 0:
                        startY = 0
                    if endY > img.shape[0] - 1:
                        endY = img.shape[0] - 1
                    rects = [startX, startY, endX, endY]
                    final_boxes[int(cls)].append(rects)
                    # boxes[path].append(xyxy2xywh(xyxy))
                    final_confidence[int(cls)].append(float(conf))
                    all_labels.append(classes[int(cls)])
                    if debug:
                        p1 = (int(x1), int(y1))
                        p2 = (int(x2), int(y2))
                        cv2.rectangle(image_write, p1, p2, color_plans[int(cls)], 2, 1)
                        cv2.putText(image_write, classes[int(cls)] + ': ' + str(round(float(conf), 2)), (int(rects[0]),
                                    int(rects[1] - 5)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, color_plans[int(cls)], 1)
    if debug:
        # cv2.imshow('img', image_write)
        # cv2.waitKey(0)
        if len(final_boxes[0]) > 0:
            os.system('mkdir -p YOLOv5_Detections')
            cv2.imwrite('YOLOv5_Detections/{}'.format(os.path.basename(image_path)), image_write)
    return final_boxes, final_confidence, all_labels


def write_to_csv(filename, data):  # writes the given data to the csv filename provided
    with open(filename, 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)
    csvfile.close()


if __name__ == '__main__':
    images = glob.glob('/home/prabodh/workspace/visiontasks/Replace_VGG_with_YOLOv5_04PD_03NMS/NO_FACETOTAL/Missed/*.jpg')
    thresh = 0.3
    nms_thresh = 0.3
    BM_count = 0
    for img_path in images:
        print(img_path)
        st = time.time()
        result_det = yolov5_prediction(img_path, threshold=thresh, nms_threshold=nms_thresh, debug=True)
        print('Detections: ', result_det)
        rects, scores, labels = result_det
        # print('Time Taken: ', time.time() - st)
        write_to_csv('YOLOv5_person_voco_client_v1_epochs_03NMS_03CONF.csv', [os.path.basename(img_path), rects, scores, labels])
        BM_STATUS = False
        # Check BM
        if len(rects[0]) > 1:
            BM_STATUS = True
        elif len(rects[1]) > 1:
            BM_STATUS = True
        else:
            person_bbx,  face_bbx = rects[0][0] if rects[0] else [], rects[1][0] if rects[1] else []

            if person_bbx and face_bbx:
                debug = False
                try:
                    image = cv2.imread(img_path)
                    (h, w) = image.shape[:2]
                    x, y, W, H = person_bbx
                    startX, startY, endX, endY = [int(x), int(y), int(W + x), int(H + y)]
                    if startX < 0:
                        startX = 0
                    if endX > image.shape[1] - 1:
                        endX = image.shape[1] - 1

                    if startY < 0:
                        startY = 0
                    if endY > image.shape[0] - 1:
                        endY = image.shape[0] - 1
                    person_bbx = [startX, startY, endX, endY]
                    x, y, W, H = face_bbx
                    startX, startY, endX, endY = [int(x), int(y), int(W + x), int(H + y)]
                    if startX < 0:
                        startX = 0
                    if endX > image.shape[1] - 1:
                        endX = image.shape[1] - 1

                    if startY < 0:
                        startY = 0
                    if endY > image.shape[0] - 1:
                        endY = image.shape[0] - 1
                    face_bbx = [startX, startY, endX, endY]
                    # Step - 3 : Calculate overlap for all boxes.
                    iou = bb_intersection_over_union(face_bbx, person_bbx)
                    print(iou)
                    BM_STATUS = iou <= 0
                    if debug:
                        print('Is face is of same person : ', BM_STATUS)
                        # cv2.imwrite('{}.jpg'.format(index), image)
                        print(face_bbx, [startX, startY, endX, endY], image.shape)
                        p1 = (int(startX), int(startY))
                        p2 = (int(endX), int(endY))
                        cv2.rectangle(image, p1, p2, (255, 255, 0), 2, 1)
                        startX, startY, endX, endY = person_bbx
                        p1 = (int(startX), int(startY))
                        p2 = (int(endX), int(endY))
                        cv2.rectangle(image, p1, p2, (255, 0, 0), 2, 1)
                        # cv2.imshow('init', image)
                        # cv2.waitKey(0)
                        os.system('mkdir -p BM_detections')
                        cv2.imwrite('BM_detections/{}_{}_{}.jpg'.format(BM_STATUS, iou, os.path.basename(img_path)), image)
                except Exception as e:
                    print(e)

        if BM_STATUS:
            BM_count += 1
        else:
            os.system('mkdir -p YOLO_BM_Missed')
            cv2.imwrite('YOLO_BM_Missed/{}'.format(os.path.basename(img_path)), cv2.imread(img_path))
    print('BM Count: ', BM_count)









