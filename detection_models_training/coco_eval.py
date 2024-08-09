import copy
import io
from contextlib import redirect_stdout

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class CocoEvaluator:
    """
    COCO Evaluator class for evaluating object detection models.
    """
    def __init__(self, coco_gt):
        """
        Initialize the CocoEvaluator with ground truth annotations.

        Args:
            coco_gt (COCO): COCO object containing ground truth annotations.
        """
        # Deep copy the ground truth COCO object to avoid modifications
        self.coco_gt = copy.deepcopy(coco_gt)

        # Initialize COCOeval object for bounding box (bbox) IoU type
        self.coco_eval = {'bbox': COCOeval(self.coco_gt, iouType='bbox')}

        self.img_ids = []
        self.eval_imgs = {'bbox': []}

    def update(self, predictions):
        """
        Update the evaluator with new predictions.

        Args:
            predictions (dict): Dictionary containing image ids as keys and prediction results as values.
        """
        # Get unique image ids from predictions
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        # Prepare the results in COCO format
        results = self.prepare(predictions)
        with redirect_stdout(io.StringIO()):
            coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()

        coco_eval = self.coco_eval['bbox']
        coco_eval.cocoDt = coco_dt
        coco_eval.params.imgIds = list(img_ids)
        img_ids, eval_imgs = evaluate(coco_eval)

        self.eval_imgs['bbox'].append(eval_imgs)

    def accumulate(self):
        """
        Accumulate the evaluation results.
        """
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        """
        Print summary of the evaluation results.
        """
        for iou_type, coco_eval in self.coco_eval.items():
            print(f"IoU metric: {iou_type}")
            coco_eval.summarize()

    def prepare(self, predictions):
        """
        Prepare results for COCO detection evaluation.

        Args:
            predictions (dict): Dictionary containing image ids as keys and prediction results as values.

        Returns:
            list: List of prepared results in COCO format.
        """
        return self.prepare_for_coco_detection(predictions)

    @staticmethod
    def prepare_for_coco_detection(predictions):
        """
        Prepare detection results for COCO evaluation.

        Args:
            predictions (dict): Dictionary containing image ids as keys and prediction results as values.

        Returns:
            list: List of prepared results in COCO format.
        """
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    """
    Convert bounding boxes from (xmin, ymin, xmax, ymax) to (xmin, ymin, width, height).

    Args:
        boxes (torch.Tensor): Tensor containing bounding boxes in (xmin, ymin, xmax, ymax) format.

    Returns:
        torch.Tensor: Tensor containing bounding boxes in (xmin, ymin, width, height) format.
    """
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def evaluate(coco_eval):
    """
    Perform evaluation and return image ids and evaluation images.

    Args:
        coco_eval (COCOeval): COCOeval object for evaluation.

    Returns:
        tuple: Tuple containing list of image ids and numpy array of evaluation images.
    """
    with redirect_stdout(io.StringIO()):
        coco_eval.evaluate()
    return coco_eval.params.imgIds, np.asarray(coco_eval.evalImgs).reshape(-1, len(coco_eval.params.areaRng),
                                                                           len(coco_eval.params.imgIds))


def convert_to_coco_api(ds):
    """
    Converts a dataset to COCO format.

    Args:
        ds (torch.utils.data.Dataset): The dataset to convert.

    Returns:
        COCO: COCO object containing the converted dataset.
    """
    coco_ds = COCO()
    # Annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()

    for img_idx in range(len(ds)):
        # Load image and target from dataset
        img, targets = ds[img_idx]
        image_id = targets["image_id"]

        # Create image dictionary
        img_dict = {
            "id": image_id,
            "height": img.shape[-2],
            "width": img.shape[-1]
        }
        dataset["images"].append(img_dict)

        # Convert bounding boxes to COCO format (x, y, width, height)
        bboxes = targets["boxes"].clone()
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()

        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {
                "image_id": image_id,
                "bbox": bboxes[i],
                "category_id": labels[i],
                "area": areas[i],
                "iscrowd": iscrowd[i],
                "id": ann_id
            }
            categories.add(labels[i])
            dataset["annotations"].append(ann)
            ann_id += 1

    # Add categories to the dataset
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(dataset):
    """
    Wrapper function to convert a dataset to COCO format.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to convert.

    Returns:
        COCO: COCO object containing the converted dataset.
    """
    return convert_to_coco_api(dataset)
