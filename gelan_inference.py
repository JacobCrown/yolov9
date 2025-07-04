import torch
import cv2
from pathlib import Path
import numpy as np

# YOLOv9 imports
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (
    LOGGER,
    check_img_size,
    non_max_suppression,
    scale_boxes,
    xyxy2xywh,
)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run_gelan_inference(
    weights_path,
    image_path,
    output_path,
    conf_thres=0.25,
    iou_thres=0.45,
    imgsz=640,
    device="",
):
    """
    Run YOLOv9 GELAN inference on a single image

    Args:
        weights_path (str): Path to the model weights (.pt file)
        image_path (str): Path to the input image
        output_path (str): Path to save the output image with detections
        conf_thres (float): Confidence threshold for detections
        iou_thres (float): IoU threshold for NMS
        imgsz (int): Inference image size
        device (str): Device to run inference on ('', 'cpu', '0', '1', etc.)
    """

    # Setup device
    device = select_device(device)
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {weights_path}")
    model = DetectMultiBackend(weights_path, device=device, dnn=False, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Load image
    print(f"Loading image from {image_path}")
    dataset = LoadImages(image_path, img_size=imgsz, stride=stride, auto=pt)

    # Warmup
    if device.type != "cpu":
        model.warmup(imgsz=(1, 3, *imgsz))

    # Process image
    for path, im, im0s, vid_cap, s in dataset:
        # Preprocess
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        print("Running inference...")
        pred = model(im, augment=False, visualize=False)

        # NMS
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, None, False, max_det=1000
        )

        # Process predictions
        for i, det in enumerate(pred):  # per image
            im0 = im0s.copy()

            # Create annotator
            annotator = Annotator(im0, line_width=3, example=str(names))

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                detection_count = {}
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    detection_count[names[int(c)]] = int(n)

                print(f"Detections found: {detection_count}")

                # Draw boxes and labels
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f"{names[c]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(c, True))
            else:
                print("No detections found")

            # Get final image
            im0 = annotator.result()

            # Save results
            print(f"Saving result to {output_path}")
            cv2.imwrite(output_path, im0)

            break  # Only process first image


def main():
    """
    Example usage of the inference function
    """
    # Configuration
    weights_path = "gelan-c.pt"  # Path to your YOLOv9 GELAN weights
    image_path = "data/images/horses.jpg"  # Path to your input image
    output_path = "output/output_gelan.jpg"  # Path for output image

    # Run inference
    run_gelan_inference(
        weights_path=weights_path,
        image_path=image_path,
        output_path=output_path,
        conf_thres=0.25,  # Confidence threshold
        iou_thres=0.45,  # IoU threshold for NMS
        imgsz=640,  # Input image size
        device="",  # '' for auto-detect, 'cpu' for CPU, '0' for GPU 0
    )

    print("Inference completed!")


if __name__ == "__main__":
    main()
