# WSI Detection and MGC/Tumor Ratio Calculation

This package contains scripts for detecting multi-nucleated giant cells (MGC) and tumor cells in whole slide images (WSIs) using a trained FCOS detection model, and for calculating the ratio of MGC to tumor cells for each WSI.

## Requirements

- Python 3.8
- pandas

Install the required packages using:

```bash
pip install pandas tqdm
```

## Scripts
### 1. WSI Detection Script

This script processes WSIs to detect MGC and tumor cells using a trained FCOS detection model.

#### Usage

```bash
python wsi_detection_model_inference.py --wsi-folder <path_to_wsi_folder> --weights <path_to_model_weights> --threshold <detection_threshold> --tile-size <tile_size> --stride <stride> --output-csv <output_csv_file>
```

#### Example

```bash
python wsi_detection_model_inference.py --wsi-folder /path/to/wsi_folder --weights model.pth --threshold 0.5 --tile-size 512 --stride 256 --output-csv detection_results.csv
```

### 2. Compute MGC/Tumor Ratio Script

This script takes two input CSV files (one with the number of detected MGCs and one with the number of detected tumor cells) and computes the ratio of MGC to tumor cells for each WSI. The results are written to a new CSV file.

##### Usage

```bash
python compute_mgc_tumor_ratio.py --mgc-csv <path_to_mgc_csv> --tumor-csv <path_to_tumor_csv> --output-csv <path_to_output_csv>
```

#### Example

```bash
python compute_mgc_tumor_ratio.py --mgc-csv mgc_detections.csv --tumor-csv tumor_detections.csv --output-csv mgc_tumor_ratio.csv
```

## Sequential Usage

To run both scripts sequentially to detect cells and compute the MGC/tumor ratio, follow these steps:

1. Run the WSI detection script for MGC:

    ```bash
    python wsi_detection_model_inference.py --wsi-folder /path/to/wsi_folder --weights mgc_model.pth --threshold 0.5 --tile-size 512 --stride 256 --output-csv mgc_detections.csv
    ```
   
2. Run the WSI detection script for tumor cells:

    ```bash
    python wsi_detection_model_inference.py --wsi-folder /path/to/wsi_folder --weights tumor_model.pth --threshold 0.5 --tile-size 512 --stride 256 --output-csv tumor_detections.csv
    ```

3. Compute the MGC/tumor ratio:

    ```bash
    python compute_mgc_tumor_ratio.py --mgc-csv mgc_detections.csv --tumor-csv tumor_detections.csv --output-csv mgc_tumor_ratio.csv
    ```

This will produce a CSV file (mgc_tumor_ratio.csv) with the ratio of MGC to tumor cells for each WSI.


### Usage Instructions

To run the scripts sequentially:

1. Detect MGC:
    ```bash
    python wsi_detection_model_inference.py --wsi-folder /path/to/wsi_folder --weights mgc_model.pth --threshold 0.5 --tile-size 512 --stride 256 --output-csv mgc_detections.csv
    ```

2. Detect tumor cells:
    ```bash
    python wsi_detection_model_inference.py --wsi-folder /path/to/wsi_folder --weights tumor_model.pth --threshold 0.5 --tile-size 512 --stride 256 --output-csv tumor_detections.csv
    ```

3. Compute the MGC/tumor ratio:
    ```bash
    python compute_mgc_tumor_ratio.py --mgc-csv mgc_detections.csv --tumor-csv tumor_detections.csv --output-csv mgc_tumor_ratio.csv
    ```

These steps will produce the final CSV file with the MGC to tumor cell ratios.
