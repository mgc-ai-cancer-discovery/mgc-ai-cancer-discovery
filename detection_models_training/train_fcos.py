"""
Train FCOS Detection Model

This script trains a Fully Convolutional One-Stage (FCOS) detection model using annotated data.
It performs the following steps:
1. Loads and preprocesses the training and validation data.
2. Initializes the FCOS model with a ResNet50 backbone pre-trained on ImageNet.
3. Applies data augmentation to the input data.
4. Trains the model using the specified hyperparameters.
5. Saves the model with the highest mean Average Precision (mAP) on the validation set.

Usage:
    python train_fcos.py --data-dir <path_to_training_data> --val-dir <path_to_validation_data> --output-dir <path_to_save_model> --batch-size <batch_size> --epochs <number_of_epochs> --lr <learning_rate> --weight-decay <weight_decay>

Example:
    python train_fcos.py --data-dir input_data --val-dir validation_data --output-dir models/mgc_model --batch-size 4 --epochs 2000 --lr 1e-4 --weight-decay 1e-4
"""

import os
import argparse
import types

import cv2
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from albumentations import Compose, Rotate, Flip, ShiftScaleRotate, ColorJitter, RandomCrop, CenterCrop
from albumentations.pytorch import ToTensorV2
from staintools import StainAugmentor
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.fcos import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights

from engine import train_one_epoch, evaluate


# Initialize the stain augmentor
stain_augmentor = StainAugmentor(method='vahadane')


def stain_augment(image):
    """
    Apply stain augmentation to the image.

    Args:
        image (np.array): The input image.

    Returns:
        np.array: The augmented image.
    """
    stain_augmentor.fit(image)
    return stain_augmentor.pop()


class CustomDataset(Dataset):
    """
    Custom dataset for loading images and annotations.
    """
    def __init__(self, image_dir, transform=None):
        """
        Initialize the dataset.

        Args:
            image_dir (str): Directory containing the images.
            transform (albumentations.Compose, optional): Data augmentation pipeline.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            tuple: Tuple containing the image and its corresponding target.
        """
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply stain augmentation
        image = stain_augment(image).astype(np.uint8)

        csv_file = image_file.replace('.png', '.csv')
        csv_path = os.path.join(self.image_dir, csv_file)
        bboxes = pd.read_csv(csv_path).values.tolist()

        if self.transform:
            augmented = self.transform(image=image, bboxes=bboxes)
            image = augmented['image']
            bboxes = augmented['bboxes']

        if len(bboxes) == 0:
            bboxes = np.empty((0, 4))
        else:
            bboxes = np.asarray(bboxes)[:, :4]

        # Values for CoCo evaluation
        image_id = idx
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        iscrowd = torch.zeros((len(bboxes),), dtype=torch.int64)

        target = {
            'boxes': torch.tensor(bboxes).int(),
            'labels': torch.tensor([0] * len(bboxes)).long(),
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }

        return image.float(), target


def main():
    """
    Main function to parse arguments and run the detection script.
    """
    # Argument parser
    parser = argparse.ArgumentParser(description="FCOS Training Script")
    parser.add_argument('--train-dir', type=str, default='input_data', help='Path to the training data directory')
    parser.add_argument('--val-dir', type=str, default='validation_data', help='Path to the validation data directory')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train for')
    parser.add_argument('--learning-rate', '--lr', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for the optimizer')
    parser.add_argument('--output-dir', type=str, default='output', help='Directory to save the trained models')
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Data augmentation
    train_transform = Compose([
        Rotate(limit=90, p=0.5),
        Flip(p=0.5),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=20, p=0.5),
        ColorJitter(p=0.5),
        RandomCrop(256, 256),
        ToTensorV2()
    ], bbox_params={'format': 'pascal_voc', 'label_fields': [], 'min_visibility': .1})
    val_transform = Compose([
        CenterCrop(256, 256),
        ToTensorV2()
    ], bbox_params={'format': 'pascal_voc', 'label_fields': [], 'min_visibility': .1})

    # Dataset and DataLoader for training
    train_dataset = CustomDataset(image_dir=args.train_dir, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # Dataset and DataLoader for validation
    val_dataset = CustomDataset(image_dir=args.val_dir, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # Model, optimizer ; the loss function is directly integrated into the torchvision model
    model = fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights)

    def no_resize(self, image, target):
        return image, target
    # Discard resizing from the original model
    model.transform.resize = types.MethodType(no_resize, model.transform.resize)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    best_map = 0
    for epoch in range(args.epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=1)
        coco_eval = evaluate(model, val_loader, device)

        # Fetch average precision @IoU 50% and save model
        map = coco_eval.coco_eval['bbox'].stats[1]
        model_save_path = os.path.join(args.output_dir, f'model_epoch{epoch}_map{map:.4f}.pth')
        torch.save(model.state_dict(), model_save_path)
        if map > best_map:
            best_map = map

    print(f'Training complete. Best mAP = {best_map}')


if __name__ == '__main__':
    main()
