import torch
import numpy as np
from typing import List, Dict, Union
import sys
from pathlib import Path
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from models.common import DetectMultiBackend
from utils.dataloaders import letterbox
from utils.general import (
    check_img_size,
    non_max_suppression,
    scale_boxes,
)
from utils.torch_utils import select_device, smart_inference_mode


class ModelHandler:
    stride: int
    names: Union[List[str], Dict[int, str]]
    pt: bool

    def __init__(self, weights_path: str, device: str = "", imgsz=640):
        """
        Initializes the YOLOv9 model for inference.

        Args:
            weights_path (str): Path to the model weights (.pt file).
            device (str): Device to run inference on ('', 'cpu', '0', '1', etc.).
            imgsz (int): Inference image size.
        """
        self.device = select_device(device)
        self.model = DetectMultiBackend(
            weights_path,
            device=self.device,
            dnn=False,
            fp16=(self.device.type != "cpu"),
        )
        self.stride, self.names, self.pt = (
            self.model.stride,
            self.model.names,
            self.model.pt,
        )
        self.imgsz = check_img_size(imgsz, s=self.stride)

        # Warmup
        if self.device.type != "cpu":
            self.model.warmup(imgsz=(1, 3, self.imgsz, self.imgsz))

    @smart_inference_mode()
    def predict(self, image: np.ndarray, conf_thres=0.25, iou_thres=0.45):
        """
        Performs inference on a single image.

        Args:
            image (np.ndarray): The input image in BGR format.
            conf_thres (float): Confidence threshold for detections.
            iou_thres (float): IoU threshold for NMS.

        Returns:
            list: A list of detection results, where each result is a dictionary.
        """
        # Preprocess image
        im = letterbox(image, self.imgsz, stride=self.stride, auto=self.pt)[0]
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)

        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = self.model(im, augment=False, visualize=False)

        # NMS - For YOLO models with dual heads, select the primary prediction
        pred = pred[0][1]
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes=None, agnostic=False, max_det=1000
        )

        results = []
        # Process predictions
        for i, det in enumerate(pred):  # per image
            if len(det):
                # Rescale boxes from img_size to original image size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], image.shape).round()

                # Format results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    result = {
                        "box": [int(coord) for coord in xyxy],
                        "confidence": float(conf),
                        "class_id": c,
                        "class_name": self.names[c],
                    }
                    results.append(result)
        return results
