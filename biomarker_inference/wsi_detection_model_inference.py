"""
WSI Detection Script

This script processes whole slide images (WSIs) using a trained FCOS detection model.
It performs the following steps for each WSI:
1. Tessellates the WSI into smaller tiles.
2. Applies tissue detection to remove background tiles.
3. Runs inference on the tiles using the trained FCOS model.
4. Applies a threshold to convert probabilities to detections.
5. Counts the number of detected boxes in the WSI.
6. Writes the number of detected boxes per WSI to a CSV file.

Usage:
    python wsi_detection_model_inference.py --wsi-folder <path_to_wsi_folder> --weights <path_to_model_weights> --threshold <detection_threshold> --tile-size <tile_size> --stride <stride> --output-csv <output_csv_file>

Example:
    python wsi_detection_model_inference.py --wsi-folder /path/to/wsi_folder --weights model.pth --threshold 0.5 --tile-size 512 --stride 256 --output-csv detection_results.csv
"""

import argparse
import os

import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from openslide import OpenSlide
from torchvision.models.detection import fcos_resnet50_fpn
from tqdm import tqdm


def load_model(weights_path):
    """
    Load the trained FCOS model.

    Args:
        weights_path (str): Path to the model weights.

    Returns:
        torch.nn.Module: The loaded model.
    """
    model = fcos_resnet50_fpn(pretrained=False)
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    return model


def infer_tile(model, tile, device, threshold):
    """
    Perform inference on a tile.

    Args:
        model (torch.nn.Module): The detection model.
        tile (np.array): The image tile.
        device (torch.device): The device to run inference on.
        threshold (float): Detection threshold.

    Returns:
        np.array: Array of bounding boxes.
    """
    transform = ToTensorV2()
    tile = transform(image=tile)["image"].to(device).float() / 255.0
    with torch.no_grad():
        outputs = model([tile])
    outputs = outputs[0]
    keep = outputs["scores"] > threshold
    boxes = outputs["boxes"][keep].cpu().numpy()
    return boxes


def contains_tissue(tile, threshold=0.8):
    """
    Check if the tile contains tissue.

    Args:
        tile (np.array): The image tile.
        threshold (float): Tissue threshold.

    Returns:
        bool: True if the tile contains tissue, else False.
    """
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tissue_ratio = 1 - (cv2.countNonZero(binary) / binary.size)
    return tissue_ratio > threshold


def process_wsi(wsi_path, model, device, threshold, tile_size, stride):
    """
    Process a whole slide image (WSI) and perform inference.

    Args:
        wsi_path (str): Path to the WSI file.
        model (torch.nn.Module): The detection model.
        device (torch.device): The device to run inference on.
        threshold (float): Detection threshold.
        tile_size (int): Size of the tiles to extract from WSI.
        stride (int): Stride for tiling WSI.

    Returns:
        list: List of bounding boxes.
    """
    slide = OpenSlide(wsi_path)
    slide_width, slide_height = slide.dimensions
    boxes = []

    for y in tqdm(range(0, slide_height, stride)):
        for x in range(0, slide_width, stride):
            tile = np.array(slide.read_region((x, y), 0, (tile_size, tile_size)))[:, :, :3]
            if contains_tissue(tile):
                tile_boxes = infer_tile(model, tile, device, threshold)
                for box in tile_boxes:
                    box[0] += x
                    box[1] += y
                    box[2] += x
                    box[3] += y
                boxes.extend(tile_boxes)
    return boxes


def process_wsi_folder(wsi_folder, model, device, threshold, tile_size, stride, output_csv):
    """
    Process all WSIs in a folder and save detection results to a CSV file.

    Args:
        wsi_folder (str): Path to the folder containing WSIs.
        model (torch.nn.Module): The detection model.
        device (torch.device): The device to run inference on.
        threshold (float): Detection threshold.
        tile_size (int): Size of the tiles to extract from WSIs.
        stride (int): Stride for tiling WSIs.
        output_csv (str): Path to the output CSV file.
    """
    wsi_files = [f for f in os.listdir(wsi_folder) if f.endswith('.svs')]
    results = []

    for wsi_file in wsi_files:
        wsi_path = os.path.join(wsi_folder, wsi_file)
        boxes = process_wsi(wsi_path, model, device, threshold, tile_size, stride)
        num_detections = len(boxes)
        results.append({'WSI': wsi_file, 'Detections': num_detections})

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)


def main():
    """
    Main function to parse arguments and run the detection script.
    """
    parser = argparse.ArgumentParser(description="WSI Detection Script")
    parser.add_argument('--wsi-folder', type=str, required=True, help='Path to the folder containing WSIs')
    parser.add_argument('--weights', type=str, required=True, help='Path to the trained model weights')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')
    parser.add_argument('--tile-size', type=int, default=512, help='Size of the tiles to extract from WSIs')
    parser.add_argument('--stride', type=int, default=256, help='Stride for tiling WSIs')
    parser.add_argument('--output-csv', type=str, default='detection_results.csv',
                        help='Output CSV file to save detection results')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.weights)
    model.to(device)

    process_wsi_folder(args.wsi_folder, model, device, args.threshold, args.tile_size, args.stride, args.output_csv)


if __name__ == "__main__":
    main()
