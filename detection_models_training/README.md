# FCOS Tumor and MGC Detection Model

This package contains the implementation of a Fully Convolutional One-Stage (FCOS) detection architecture for detecting tumor cells and MGC (multi-nucleated giant cells) in images. The model is trained and validated using PyTorch and follows the described methodology, including data augmentation and model evaluation using mean Average Precision (mAP).

## Requirements
- Python 3.8
- PyTorch 1.13
- torchvision 0.14.0
- OpenCV
- Albumentations 1.2.1
- pycocotools
- stainTools

### Installation
Create a virtual environment for python3.8 using:

```bash
virtualenv venv -p python3.8
source venv/bin/activate
```

Install the required packages using:

```bash
pip install -r requirements.txt
```

If you stumble upon an error for the installation of `spams`, ensure you have all the requirements installed, cf [the official repository](https://pypi.org/project/spams/).

## Dataset
Place all your images and their corresponding CSV files in the input_data folder. Each image file should have a corresponding CSV file with the same name (e.g., example.png and example.csv).

### CSV Format
Each CSV file should contain bounding box coordinates in the following format:

```yaml
x_min,y_min,x_max,y_max
1671,1244,1946,1576
```

## Model Architecture
The model uses a ResNet50 backbone pre-trained on ImageNet. It includes classification and regression heads, each comprising four ReLU-activated convolutional layers with batch normalization.

## Data Augmentation
Data augmentation is applied using the Albumentations and stainTools libraries. The augmentations include rotations, flips, shifts, color jitter, and stain augmentation.

## Training
The model is trained using the Adam optimizer with a learning rate of 1e-4 and a weight decay of 1e-4. The training runs for up to 2000 epochs, and the model snapshot with the highest mAP@50 on the validation set is saved.

## Usage
### Training
To train the model with the parameters used in the paper, run the following script:

```bash
python train_fcos.py
```

To train the model with specific parameters:

```bash
python train_fcos.py --train-dir <path_to_training_data> --val-dir <path_to_validation_data> --batch-size <batch_size> --epochs <number_of_epochs> --learning-rate <learning_rate> --weight-decay <weight_decay> --output-dir <output_directory>
```

#### Example

```bash
python train_fcos.py --train-dir input_data --val-dir validation_data --batch-size 2 --epochs 2000 --learning-rate 1e-4 --weight-decay 1e-4 --output-dir output
```


### Evaluation
To evaluate the model, the validate function computes the mAP@50 using the COCO evaluation metrics. The function processes the model's outputs and ground truth annotations, converting them to COCO format, and then evaluates the detections using the pycocotools library.

## Notes

* The criterion in the training loop is a placeholder for the actual focal loss used in the original implementation.
* The detection processing logic in the validate function needs to be adapted to match the model's output format.
* Ensure that the validation_data directory is structured similarly to the input_data directory with images and corresponding CSV files.

## Acknowledgments

* This implementation uses the ResNet50 architecture pre-trained on ImageNet.
* Data augmentation is performed using the Albumentations library.
