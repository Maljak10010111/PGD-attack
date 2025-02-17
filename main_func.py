# README how to run the code

import json
import torch
import argparse
from ingredients.models import get_local_cifar10_model
from ingredients.dataset import get_dataset_loaders
from ingredients.utilities import set_seed
# from PGD_last_adv_example import PGD_Attack
# from PGD_best_adv_example import PGD_Attack
from PGD_last_example_using_optim_and_sched import PGD_Attack
import matplotlib.pyplot as plt
import numpy as np


def read_config_file(config_file_path):
    with open(config_file_path, "r") as config_file:
        config_data = json.load(config_file)
    return config_data


def accuracy(model, data_loader, device, pgd_attack, num_samples):
    model.eval()

    num_correct = 0

    if pgd_attack is not None:
        for i, data in enumerate(data_loader, 0):
            images, labels = data

            images = images.to(device)
            labels = labels.to(device)

            manipulated_images = pgd_attack(images, labels)

            outputs = model(manipulated_images)
            predictions = torch.argmax(outputs, 1)
            num_correct += (predictions == labels).sum().item()
    else:
        for i, data in enumerate(data_loader, 0):
            images, labels = data

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, 1)
            num_correct += (predictions == labels).sum().item()


    return num_correct / num_samples


def tensor_to_image(tensor):
    image = tensor.cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    return image


def visualize_images(dataloader, device, pgd_attack, batch_size):
    if batch_size == 1:
        for k, data in enumerate(dataloader, 0):
            images, labels = data

            images = images.to(device)
            labels = labels.to(device)
            adversarial_images = pgd_attack(images, labels).to(device)

            for i in range(batch_size):
                img = tensor_to_image(adversarial_images[i])
                plt.figure(figsize=(2, 2))
                plt.imshow(img)
                plt.axis('off')
                plt.tight_layout()
                plt.show()
    else:
        for k, data in enumerate(dataloader, 0):
            images, labels = data

            images = images.to(device)
            labels = labels.to(device)
            adversarial_images = pgd_attack(images, labels).to(device)

            for i in range(len(adversarial_images)):
                img = tensor_to_image(adversarial_images[i])
                plt.figure(figsize=(2, 2))
                plt.imshow(img)
                plt.axis('off')
                plt.tight_layout()
                plt.title(f"Image {i + 1} in batch {k + 1}")
                plt.show()


def execute():
    parser = argparse.ArgumentParser(description='Run experiments based on a configuration file.')
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    parser.add_argument('--device', default='cpu', type=str, help='Device to use for computation')

    args = parser.parse_args()

    config_file_path = args.config
    config_data = read_config_file(config_file_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Trying computations on {device}")


    for experiment in config_data["experiments"]:
        set_seed(config_data["seed"])

        dataset = experiment["dataset"]
        model_name = experiment["model"]
        batch_size = experiment["batch_size"]
        n_samples = experiment["n_samples"]
        attack_name = experiment["attack_name"]
        attack_steps = experiment["attack_steps"]
        epsilon = experiment["epsilon"]
        optimizer = experiment["optimizer"]
        scheduler = experiment["scheduler"]


        dataloader = get_dataset_loaders(dataset, batch_size=batch_size, n_examples=n_samples, seed=config_data["seed"])

        model = get_local_cifar10_model(model_name)

        model = model.to(device)

        if PGD_Attack.__module__ == "PGD_last_adv_example" or PGD_Attack.__module__ == "PGD_best_adv_example":
            pgd_attack = PGD_Attack(model, device, epsilon=epsilon, iterations=attack_steps)
        else:
            pgd_attack = PGD_Attack(model, device, epsilon=epsilon, iterations=attack_steps, optimizer=optimizer, scheduler=scheduler)

        clean_accuracy = accuracy(model, dataloader, device, None, n_samples)
        print(f'Clean Accuracy of {model_name} model: ', clean_accuracy * 100)

        accuracy_after_attack = accuracy(model, dataloader, device, pgd_attack, n_samples)
        print(f"Accuracy after {attack_name} attack on {model_name} model: ", accuracy_after_attack * 100)

        # visualize_images(dataloader, device, pgd_attack, batch_size)


if __name__ == "__main__":
    execute()