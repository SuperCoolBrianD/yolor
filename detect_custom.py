# import argparse
# import os
# import platform
# import shutil
# import time
# from pathlib import Path
#
# import cv2
# import torch
import torch.backends.cudnn as cudnn
from numpy import random
# from utils.datasets import letterbox
# from utils.google_utils import attempt_load
# from utils.datasets import LoadStreams, LoadImages
# from utils.general import (
#     check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.models import *
from utils.datasets import *
from utils.general import *


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


def init_yoloR(weights='yolor_p6.pt', cfg='cfg/yolor_p6.cfg', names='data/coco.names', out='inference/output', imgsz=1280, half=True):
    device = select_device()
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    # half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    model = Darknet(cfg, imgsz).cuda()
    model.load_state_dict(torch.load(weights, map_location=device)['model'])
    #model = attempt_load(weights, map_location=device)  # load FP32 model
    #imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    names = load_classes(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    cudnn.benchmark = True
    return model, device, colors, names


def detect(source, device, model, colors, names, half=True, view_img=True, imgsz=1280):
    t0 = time.time()
    # dataset = LoadImages(source, img_size=imgsz, auto_size=64)
    im0s = source.copy()

    # Run inference

    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    # im0 = None
    detections = []



    img = letterbox(source, new_shape=imgsz, auto_size=64)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False)
    t2 = time_synchronized()

    for i, det in enumerate(pred):  # detections per image
        im0 = im0s

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in det:
                label = '%s %.2f' % (names[int(cls)], conf)
                if names[int(cls)] in ['car', 'person', 'truck', 'bus']:
                    detections. append([[int(i) for i in xyxy], names[int(cls)], float(conf)])
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

        # Print time (inference + NMS)

        # Stream results
        if view_img:
            cv2.imshow('0', im0)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration
    if view_img:
        cv2.waitKey(0)

    # print('Camera Detection Time. (%.3fs)' % (time.time() - t0))
    return im0, detections

if __name__ == '__main__':

    with torch.no_grad():
        img = cv2.imread('inference/images/horses.jpg')
        model, device, colors, names = init_yoloR()
        detect(img, model=model, device=device, colors=colors, names=names)
