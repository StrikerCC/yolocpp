# coding:utf-8

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 12/7/21 11:20 AM
"""

import os
import sys
# from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from utils.plots import Annotator, colors, save_one_box
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from models.common import DetectMultiBackend

from utils.torch_utils import select_device, time_sync

from dataset_patch import LoadImg2PatchBatch


class Yolo:
    def __init__(self):
        self.id = 'yolov5s'
        self.weights = '../weights/last.pt'
        self.dataset = LoadImg2PatchBatch()
        self.device = 0
        self.half = True

    # def detect(self):
    #     detections = []
    #     for i in range(10):
    #         detection = []
    #         for j in range(5):
    #             detection.append(j)
    #         detections.append(detection)
    #     return detections

    def detect(self):
        imgsz = 640

        source = '/home/cheng/proj/data/data_matrix/no/img/13/'
        sources = os.listdir(source)
        for src in sources:
            imgs = cv2.imread(source + src)

            self.dataset.load(imgs)

            conf_thres = 0.25  # confidence threshold
            iou_thres = 0.45  # NMS IOU threshold
            max_det = 1000  # maximum detections per image

            results = []

            # Load model
            device = select_device(self.device)
            model = DetectMultiBackend(self.weights, device=device)
            stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
            # imgsz = check_img_size(imgsz, s=stride)  # check image size

            # half
            if pt:
                model.model.half() if self.half else model.model.float()

            # dataloader
            dataset = self.dataset
            # dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)

            # run predication
            dt, seen = [0.0, 0.0, 0.0], 0
            for ims, ims0, img, origins in dataset:
                t1 = time_sync()
                ims = torch.from_numpy(ims).to(device)
                ims = ims.half() if self.half else ims.float()  # uint8 to fp16/32
                ims /= 255  # 0 - 255 to 0.0 - 1.0
                if len(ims.shape) == 3:
                    ims = ims[None]  # expand for batch dim
                t2 = time_sync()
                # dt[0] += t2 - t1

                # Inference
                pred = model(ims)
                t3 = time_sync()
                # dt[1] += t3 - t2

                # NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)
                # dt[2] += time_sync() - t3

                # Process predictions

                for i, det in enumerate(pred):  # per image
                    seen += 1
                    im0 = ims0[i]
                    img0 = img
                    origin = origins[i]
                    annotator = Annotator(np.ascontiguousarray(im0), line_width=3, example=str(names))

                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(ims.shape[2:], det[:, :4], im0.shape).round()
                        # det[:, :2] += torch.from_numpy(origin[::-1])
                        # det[:, 2:4] += torch.from_numpy(origin[::-1])
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            # if save_txt:  # Write to file
                            #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            #     line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            #     with open(txt_path + '.txt', 'a') as f:
                            #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            # if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = f'{names[c]} {conf:.2f}'
                            print(xyxy, label)
                            annotator.box_label(xyxy, label, color=colors(c, True))

                    results.append(det.tolist())
                    # Print time (inference-only)
                    # LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

                    # Stream results
                    im0 = annotator.result()
                    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
                    cv2.imshow("img", im0)
                    cv2.waitKey(0)  # 1 millisecond

            print(len(results), len(results[0]))
            print(results)
        return results


def main():
    det = Yolo()
    result = det.detect()
    # for batch in result:
    #     imgs, origins = batch
    #     for img, origin in zip(imgs, origins):
    #         img = img[::-1].transpose((1, 2, 0))
    #
    #         cv2.imshow('img', img)
    #         cv2.waitKey(1000)
    #         print(origin)


if __name__ == '__main__':
    main()

