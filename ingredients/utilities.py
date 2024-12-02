import datetime
import os
import re
import statistics
import warnings
import random
import numpy as np
from torchvision import utils as vutils
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F


def set_seed(seed):
    """ Random seed generation for PyTorch. See https://pytorch.org/docs/stable/notes/randomness.html
        for further details.
    Args:
        seed (int): the seed for pseudonumber generation.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def run_attack(model: nn.Module,
               loader: DataLoader,
               attack,
               ) -> dict:
    # code adapted from Official adversarial library repo:
    # https://github.com/jeromerony/adversarial-library/blob/main/adv_lib/utils/attack_utils.py

    torch.cuda.empty_cache()
    model.eval()
    device = next(model.parameters()).device
    loader_length = len(loader)

    if device.type == 'cuda':
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    else:
        start, end = 0, 0

    true_labels, pred_labels_adv = [], []
    all_inputs, all_adv_examples = [], []
    times = []

    for i, (inputs, labels) in enumerate(tqdm(loader, ncols=80, total=loader_length)):
        true_labels.append(labels.cpu().tolist())

        all_inputs.append(inputs)

        inputs, labels = inputs.to(device), labels.to(device)

        torch.cuda.empty_cache()

        if device.type == 'cuda':
            start.record()
            torch.cuda.reset_peak_memory_stats(device=device)
        try:
            adv_inputs = attack(model, inputs, labels)
        except Exception as e:
            adv_inputs = inputs
            if 'out of memory' in str(e) or 'valid cuDNN' in str(e):
                print('\n WARNING: ran out of memory, cannot perform this specific attack with this batch size')
                exit()
            else:
                raise e

        torch.cuda.empty_cache()
        elapsed_time = 0
        if device.type == 'cuda':
            end.record()
            torch.cuda.synchronize()
            elapsed_time = (start.elapsed_time(end)) / 1000
        times.append(elapsed_time)

        if adv_inputs.min() < 0 or adv_inputs.max() > 1:
            warnings.warn('Values of produced adversarials are not in the [0, 1] range -> Clipping to [0, 1].')
            adv_inputs.clamp_(min=0, max=1)

        adv_logits = model(adv_inputs)
        adv_pred = adv_logits.argmax(dim=1)
        pred_labels_adv.append(adv_pred.cpu().tolist())
        all_adv_examples.append(adv_inputs)

    data = {
        'true_labels': [item for sublist in true_labels for item in sublist],
        'pred_labels_adv': [item for sublist in pred_labels_adv for item in sublist],
        'times': times
    }

    if len(all_inputs) > 1:
        all_inputs = torch.cat(all_inputs, dim=0)
        all_adv_examples = torch.cat(all_adv_examples, dim=0)
    data['inputs'] = all_inputs
    data['adv_examples'] = all_adv_examples

    return data


def run_predictions(model: nn.Module, clean_dataset: DataLoader, adv_dataset: DataLoader,
                    rejection_class: int = None) -> dict:
    torch.cuda.empty_cache()
    model.eval()
    device = next(model.parameters()).device
    dataset_length = len(clean_dataset)

    ori_success, rejections_on_clean_dataset, rejections_on_adv_dataset = [], [], []
    true_labels, pred_labels, adv_pred_labels = [], [], []

    for i, ((inputs, labels), (adv_inputs, adv_labels)) in enumerate(
            tqdm(zip(clean_dataset, adv_dataset), ncols=80, total=dataset_length)):
        true_labels.extend(labels.cpu().tolist())

        inputs, labels = inputs.to(device), labels.to(device)
        adv_inputs = adv_inputs.to(device)

        try:
            logits = model(inputs)
            adv_logits = model(adv_inputs)
        except RuntimeError as e:
            if 'out of memory' in str(e) or 'valid cuDNN' in str(e):
                print('\n WARNING: ran out of memory, cannot perform experiments with this batch size')
                raise e
            else:
                raise e
        torch.cuda.empty_cache()

        predictions = logits.argmax(dim=1)
        adv_predictions = adv_logits.argmax(dim=1)

        pred_labels.extend(predictions.cpu().tolist())
        adv_pred_labels.extend(adv_predictions.cpu().tolist())

        if rejection_class:
            rejections_on_clean_dataset.extend((predictions == rejection_class).cpu().tolist())
            rejections_on_adv_dataset.extend((adv_predictions == rejection_class).cpu().tolist())

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if rejection_class:
        clean_accuracy = sum([pred_labels[i] == true_labels[i] for i in range(len(true_labels))]) / len(true_labels)
        adv_acc = sum([adv_pred_labels[i] == true_labels[i] or adv_pred_labels[i] == rejection_class for i in
                       range(len(true_labels))]) / len(true_labels)
    else:
        clean_accuracy = sum([pred_labels[i] == true_labels[i] for i in range(len(true_labels))]) / len(true_labels)
        adv_acc = sum([adv_pred_labels[i] == true_labels[i] for i in range(len(true_labels))]) / len(true_labels)

    data = {
        'clean_accuracy': clean_accuracy,
        'adv_accuracy': adv_acc,
        'rejection_ratio_on_clean_samples': sum(rejections_on_clean_dataset) / len(rejections_on_clean_dataset) if len(
            rejections_on_clean_dataset) > 0 else 0,
        'rejection_ratio_on_adv_examples': sum(rejections_on_adv_dataset) / len(rejections_on_adv_dataset) if len(
            rejections_on_adv_dataset) > 0 else 0,
        'asr': 1 - adv_acc,
        'ori_success': ori_success,
        'true_labels': true_labels,
        'pred_labels': pred_labels,
        'adv_labels': adv_pred_labels
    }
    return data


def resize_cifar10_img(batch_tensor):
    resized_images = []

    for image_tensor in batch_tensor:
        image_tensor = image_tensor.permute(1, 2, 0)

        resized_image = F.interpolate(image_tensor.unsqueeze(0).permute(0, 3, 1, 2), size=(224, 224), mode='bilinear',
                                      align_corners=False)

        resized_image = resized_image.squeeze(0).permute(1, 2, 0)

        resized_images.append(resized_image.permute(2, 0, 1))

    return torch.stack(resized_images)
