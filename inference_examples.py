import os
import glob
from pathlib import Path
import torch

# Import inference functions
from yolo_inference import run_yolo_inference
from gelan_inference import run_gelan_inference


def run_yolo_examples():
    """
    Examples for YOLO models (e.g., yolov9-c, yolov9-e)
    """
    print("==============================")
    print("=== Running YOLO Model Examples ===")
    print("==============================")

    weights_path = "yolov9-c.pt"  # Path to YOLOv9-C/E/M/S weights
    input_dir = "data/images"
    output_dir = "output/yolo_examples"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(weights_path):
        print(f"Warning: YOLO weights not found at {weights_path}")
        print("Please download yolov9-c.pt or other YOLO models.")
        return

    # Example 1: Process a single image
    print("\n--- YOLO Example 1: Single Image ---")
    image_path = os.path.join(input_dir, "horses.jpg")
    output_path = os.path.join(output_dir, "horses_yolo_detected.jpg")
    if os.path.exists(image_path):
        run_yolo_inference(
            weights_path=weights_path, image_path=image_path, output_path=output_path
        )
    else:
        print(f"Image not found: {image_path}")


def run_gelan_examples():
    """
    Examples for GELAN models
    """
    print("\n===============================")
    print("=== Running GELAN Model Examples ===")
    print("===============================")

    weights_path = "gelan-c.pt"  # Path to GELAN weights
    input_dir = "data/images"
    output_dir = "output/gelan_examples"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(weights_path):
        print(f"Warning: GELAN weights not found at {weights_path}")
        print("Please provide GELAN model weights (e.g., gelan-c.pt).")
        return

    # Example 1: Process a single image
    print("\n--- GELAN Example 1: Single Image ---")
    image_path = os.path.join(input_dir, "bus.jpg")
    output_path = os.path.join(output_dir, "bus_gelan_detected.jpg")
    if os.path.exists(image_path):
        run_gelan_inference(
            weights_path=weights_path, image_path=image_path, output_path=output_path
        )
    else:
        print(f"Image not found: {image_path}")


def main():
    """
    Main function demonstrating various inference scenarios
    """
    # Create a main output directory
    os.makedirs("output", exist_ok=True)

    # Run YOLO examples
    try:
        run_yolo_examples()
    except Exception as e:
        print(f"An error occurred during YOLO examples: {e}")

    # Run GELAN examples
    try:
        run_gelan_examples()
    except Exception as e:
        print(f"An error occurred during GELAN examples: {e}")

    print("\n=== All examples completed! ===")


if __name__ == "__main__":
    main()
