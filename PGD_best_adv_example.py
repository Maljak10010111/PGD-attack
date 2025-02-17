import torch
from torch import nn


class PGD_Attack(nn.Module):
    def __init__(self, model, device, epsilon, iterations):
        super(PGD_Attack, self).__init__()
        self.epsilon = epsilon / 255
        self.lr = self.epsilon / 4
        self.attack_steps = iterations
        self.model = model
        self.device = device

    def forward(self, clean_images, true_labels):

        manipulated_images = torch.clone(clean_images)

        best_adv_example = torch.clone(clean_images)

        max_loss = float('-inf')

        loss_fn = nn.CrossEntropyLoss()

        for _ in range(self.attack_steps):
            manipulated_images.requires_grad = True

            self.model.zero_grad()

            logits = self.model(manipulated_images).to(self.device)

            loss = loss_fn(logits, true_labels)

            loss.backward()

            grad = manipulated_images.grad.sign()

            manipulated_images = manipulated_images + self.lr * grad

            perturbation = manipulated_images - clean_images

            perturbation = torch.clamp(perturbation, min=-self.epsilon, max=self.epsilon)

            manipulated_images = clean_images + perturbation

            manipulated_images = torch.clamp(manipulated_images, min=0, max=1)

            manipulated_images = manipulated_images.detach()

            if loss.item() > max_loss:
                max_loss = loss.item()
                best_adv_example = torch.clone(manipulated_images)

        return best_adv_example