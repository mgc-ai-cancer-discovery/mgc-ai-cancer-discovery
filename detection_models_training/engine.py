import math
import sys
import time
import torch
import utils
from coco_eval import CocoEvaluator, get_coco_api_from_dataset


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    """
    Train the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        data_loader (torch.utils.data.DataLoader): The data loader for training data.
        device (torch.device): The device to train on.
        epoch (int): The current epoch number.
        print_freq (int): The frequency of printing log messages.
        scaler (torch.cuda.amp.GradScaler, optional): The gradient scaler for mixed precision training.
    """
    # Set model to training mode
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("learning_rate", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    # Iterate over data
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        # Move images and targets to the specified device
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        # Perform forward pass with automatic mixed precision if scaler is provided
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values())

        # Reduce losses across all GPUs for logging
        reduced_loss_dict = loss_dict
        reduced_total_loss = sum(loss for loss in reduced_loss_dict.values())

        loss_value = reduced_total_loss.item()

        # Stop training if loss is not finite
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(reduced_loss_dict)
            sys.exit(1)

        # Backpropagation
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        # Update the metrics
        metric_logger.update(loss=reduced_total_loss, **reduced_loss_dict)
        metric_logger.update(learning_rate=optimizer.param_groups[0]["lr"])

    return metric_logger


@torch.inference_mode()
def evaluate(model, data_loader, device):
    """
    Evaluate the model on the validation set.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): The data loader for validation data.
        device (torch.device): The device to evaluate on.
    """
    # Set number of threads for evaluation
    num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco_api = get_coco_api_from_dataset(data_loader.dataset)
    coco_evaluator = CocoEvaluator(coco_api)

    # Iterate over validation data
    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = [img.to(device) for img in images]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - start_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        eval_start_time = time.time()
        coco_evaluator.update(res)
        eval_time = time.time() - eval_start_time

        metric_logger.update(model_time=model_time, evaluator_time=eval_time)

    print("Averaged stats:", metric_logger)

    # Accumulate predictions and summarize results
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(num_threads)
    return coco_evaluator
