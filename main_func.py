# README how to run the project

import json
from torchvision import transforms
import torch
from tqdm import tqdm
import argparse
from ingredients.models import get_clip_model, get_local_model
from ingredients.dataset import get_dataset_loaders, get_label_names
from models.MultiShield import MultiShield
from models.CLIP import ClipModel
from modified_autoattack import AutoAttack
import csv
from ingredients.utilities import run_attack, run_predictions, set_seed, resize_cifar10_img
import torch.utils.data as data_utils
from functools import partial
import os


def read_config_file(config_file_path):
    with open(config_file_path, "r") as config_file:
        config_data = json.load(config_file)
    return config_data


def auto_attack(model, image, label, device, epsilon=None, verbose=False):
    if epsilon is None:
        epsilon = 8 / 255
    x_test, y_test = image, label
    adversary = AutoAttack(model, rejection_class_index=10, norm='Linf', eps=epsilon / 255, version='custom',
                           attacks_to_run=['apgd-ce', 'apgd-dlr'], verbose=verbose, device=device)
    adversary.apgd.n_restarts = 1
    adversarial_examples = adversary.run_standard_evaluation(x_test, y_test)
    return adversarial_examples


def auto_attack_with_rejection(model, image, label, device, rejection_class_index):
    x_test, y_test = image, label
    adversary = AutoAttack(model, rejection_class_index=rejection_class_index, norm='Linf', eps=8 / 255,
                           version='custom', attacks_to_run=['apgd-ce-rejection', 'apgd-dlr-rejection'], device=device)
    adversary.apgd.n_restarts = 1
    adversarial_examples = adversary.run_standard_evaluation(x_test, y_test)
    return adversarial_examples


def creating_csv(dnn_clean_acc, dnn_robust_acc, ms_clean_acc, ms_adv_acc_on_dnn_adv_exmp, ms_adv_acc,
                 rejection_ratio_ms_on_clean_dataset,
                 rejection_ratio_ms_on_dnn_adv_dataset, rejection_ratio_ms_on_ms_adv_dataset, model_name, n_samples,
                 clip_accuracy):
    result_file = f"results_{model_name}_on_{n_samples}_samples.csv"

    final_result = os.path.join("results", result_file)

    first_row = [
        f"{model_name} Clean Accuracy",
        f"{model_name} Robust Accuracy",
        "CLIP Clean Accuracy",
        "Multi-Shield Clean Accuracy",
        "Multi-Shield Rejection Ratio on Clean Dataset"
    ]

    first_row_values = [
        dnn_clean_acc,
        dnn_robust_acc,
        clip_accuracy,
        ms_clean_acc,
        rejection_ratio_ms_on_clean_dataset
    ]

    second_row = [
        "Multi-Shield Robust Accuracy",
        "Multi-Shield Rejection Ratio on Adversarial Dataset",
        "Multi-Shield Robust Accuracy under Adaptive Attack",
        "Multi-Shield Rejection Ratio on Adversarial Dataset (under Adaptive Attack)"
    ]

    second_row_values = [
        ms_adv_acc_on_dnn_adv_exmp,
        rejection_ratio_ms_on_dnn_adv_dataset,
        ms_adv_acc,
        rejection_ratio_ms_on_ms_adv_dataset
    ]

    with open(final_result, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(first_row)
        writer.writerow(first_row_values)
        writer.writerow(second_row)
        writer.writerow(second_row_values)


def clip_accuracy(clip_all_predictions):
    all_clip_preds = torch.cat(clip_all_predictions)
    overall_clip_accuracy = all_clip_preds.mean().item()
    return overall_clip_accuracy


def forward_phase():
    parser = argparse.ArgumentParser(description='Run experiments based on a configuration file.')
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    parser.add_argument('--device', default='cpu', type=str, help='Device to use for computation')

    args = parser.parse_args()

    config_file_path = args.config
    config_data = read_config_file(config_file_path)

    device = torch.device(args.device)
    print(f"Trying computations on {device}")

    for experiment in config_data["experiments"]:
        set_seed(config_data["seed"])

        clip_model_id = experiment["clip_model_id"]
        dataset = experiment["dataset"]
        model_name = experiment["model"]
        batch_size = experiment["batch_size"]
        n_samples = experiment["n_samples"]

        label_names = get_label_names(dataset)
        rejection_class_index = len(label_names)
        dataloaders = get_dataset_loaders(dataset, batch_size=batch_size, n_examples=n_samples,
                                          seed=config_data["seed"])
        model = get_local_model(model_name, dataset)
        model.eval()
        model = model.to(device)

        images_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))

        clip_model_name, processor_name, tokenizer_name, use_open_clip = get_clip_model(clip_model_id)
        clip_model = ClipModel(clip_model_name, processor_name, tokenizer_name, use_open_clip, label_names,
                               torch_preprocess=images_normalize, dataset=dataset, device=device)
        multi_shield = MultiShield(dnn=model, clip_model=clip_model, dataset=dataset)

        attack_results_dnn = run_attack(model, dataloaders["val"], partial(auto_attack, **{"device": device}))

        adv_dataset_dnn = data_utils.TensorDataset(torch.tensor(attack_results_dnn['adv_examples']),
                                                   torch.tensor(attack_results_dnn['true_labels']))
        adv_loader_dnn = data_utils.DataLoader(adv_dataset_dnn, batch_size=batch_size, shuffle=False)

        dnn_accuracy = run_predictions(model, dataloaders["val"], adv_loader_dnn)

        ms_acc_on_dnn_adv_examples = run_predictions(multi_shield, dataloaders["val"], adv_loader_dnn,
                                                     rejection_class=rejection_class_index)

        attack_params = {"rejection_class_index": rejection_class_index, "device": device}
        attack_results_multi_shield = run_attack(multi_shield, dataloaders["val"],
                                                 partial(auto_attack_with_rejection, **attack_params))

        adv_dataset_ms = data_utils.TensorDataset(torch.tensor(attack_results_multi_shield['adv_examples']),
                                                  torch.tensor(attack_results_multi_shield['true_labels']))
        adv_loader_ms = data_utils.DataLoader(adv_dataset_ms, batch_size=batch_size, shuffle=False)

        multi_shield_acc = run_predictions(multi_shield, dataloaders["val"], adv_loader_ms,
                                           rejection_class=rejection_class_index)

        all_clip_predictions = []

        for i, (inputs, labels) in enumerate(tqdm(dataloaders["val"], ncols=80, total=len(dataloaders["val"]))):
            inputs = inputs.to(device)
            labels = labels.to(device)

            if dataset == "cifar10":
                resized_img = resize_cifar10_img(inputs)
            else:
                resized_img = inputs

            image_emb_by_clip = clip_model.create_img_emb(resized_img)

            clip_predictions = clip_model.clip_prediction(image_emb_by_clip, labels)
            all_clip_predictions.append(clip_predictions)

        print("Experimental results:")
        print(f"\n{model_name} Clean Accuracy: ", dnn_accuracy['clean_accuracy'])
        print(f"\n{model_name} Robust Accuracy:", dnn_accuracy['adv_accuracy'])
        print("\nMulti-Shield Clean Accuracy: ", multi_shield_acc['clean_accuracy'])
        print("\nMulti-Shield Robust Accuracy: ", ms_acc_on_dnn_adv_examples['adv_accuracy'])
        print("\nMulti-Shield Robust Accuracy under Adaptive Attack: ", multi_shield_acc['adv_accuracy'])
        print("\nCLIP Clean Accuracy: ", clip_accuracy(all_clip_predictions))

        creating_csv(dnn_accuracy['clean_accuracy'], dnn_accuracy['adv_accuracy'], multi_shield_acc['clean_accuracy'],
                     ms_acc_on_dnn_adv_examples['adv_accuracy'],
                     multi_shield_acc['adv_accuracy'], multi_shield_acc['rejection_ratio_on_clean_samples'],
                     ms_acc_on_dnn_adv_examples['rejection_ratio_on_adv_examples'],
                     multi_shield_acc['rejection_ratio_on_adv_examples'], model_name, n_samples,
                     clip_accuracy(all_clip_predictions))


if __name__ == "__main__":
    forward_phase()
