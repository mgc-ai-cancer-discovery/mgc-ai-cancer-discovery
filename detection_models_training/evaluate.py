"""
Evaluate FCOS Detection Model

This script evaluates a trained FCOS detection model on a test dataset.
It performs the following steps:
1. Loads the trained model weights.
2. Loads the test data from the specified directory.
3. Runs inference on the test data.
4. Computes the average precision using COCO evaluation metrics.
5. Prints the evaluation results.

Usage:
    python evaluate.py --test-dir <path_to_test_data> --weights <path_to_model_weights> --batch-size <batch_size>

Example:
    python evaluate.py --test-dir test_data --weights models/mgc_model.pth --batch-size 2
"""

import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights

from detection_models_training.train_fcos import CustomDataset
from engine import evaluate


def evaluate_model(test_image_dir, weights_path, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights)
    model.load_state_dict(torch.load(weights_path))
    model.to(device)

    # Load test data
    test_dataset = CustomDataset(image_dir=test_image_dir)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # Evaluate the model
    coco_evaluator = evaluate(model, test_loader, device)

    # Print results
    print(f"Evaluation results: {coco_evaluator.coco_eval['bbox'].stats}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate FCOS Detection Model")
    parser.add_argument('--test-dir', type=str, required=True, help='Path to the test data directory')
    parser.add_argument('--weights', type=str, required=True, help='Path to the trained model weights')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size for evaluation')
    args = parser.parse_args()

    evaluate_model(args.test_dir, args.weights, args.batch_size)


if __name__ == "__main__":
    main()
