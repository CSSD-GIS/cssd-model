import glob
import os
import time
from io import BytesIO

import torch
from PIL import Image
from vizer.draw import draw_boxes

from ssd.config import cfg
from ssd.data.datasets import COCODataset, VOCDataset
import argparse
import numpy as np
import cv2 as cv

from ssd.data.transforms import build_transforms
from ssd.modeling.detector import build_detection_model
from ssd.utils import mkdir
from ssd.utils.checkpoint import CheckPointer
from concurrent import futures

import grpc

import detection_proto_pb2
import detection_proto_pb2_grpc

model = None
transforms =  None

class Detection(detection_proto_pb2_grpc.DetectionServicer):
    def Predict(self, request, context):
        detected_image = predict(request.originImage)
        print("检测完毕.")
        return detection_proto_pb2.DetectionResponse(predictedImage = detected_image)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    detection_proto_pb2_grpc.add_DetectionServicer_to_server(Detection(), server)
    server.add_insecure_port('[::]:9001')
    server.start()
    server.wait_for_termination()

@torch.no_grad()
def load_model(cfg, ckpt):
    model = build_detection_model(cfg)
    model = model.to("cpu")
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
    print('Loaded weights from {}'.format(weight_file))

    transforms = build_transforms(cfg, is_train=False)
    model.eval()

    return model, transforms


def init():
    global model, transforms
    config_file = "./configs/config512.yaml"
    cfg.merge_from_file(config_file)
    cfg.freeze()
    print("Loaded configuration file {}".format(config_file))

    model, transforms = load_model(cfg=cfg, ckpt=None)


@torch.no_grad()
def predict(image_bytes):
    global model, transforms
    class_names = VOCDataset.class_names

    start = time.time()
    bytes_stream = BytesIO(image_bytes)
    image = np.array(Image.open(bytes_stream).convert("RGB"))

    height, width = image.shape[:2]
    images = transforms(image)[0].unsqueeze(0)
    load_time = time.time() - start

    start = time.time()
    result = model(images.to("cpu"))[0]
    inference_time = time.time() - start

    result = result.resize((width, height)).to(torch.device("cpu")).numpy()
    boxes, labels, scores = result['boxes'], result['labels'], result['scores']

    indices = scores > 0.7
    boxes = boxes[indices]
    labels = labels[indices]
    scores = scores[indices]

    drawn_image = draw_boxes(image, boxes, labels, scores, class_names).astype(np.uint8)
    Image.fromarray(drawn_image).save(os.path.join("D:/", "1234.jpg"))
    return drawn_image.tobytes()


if __name__ == '__main__':
    init()
    serve()
