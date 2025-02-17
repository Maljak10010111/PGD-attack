import torch
from torch import nn
from torch import optim


def get_optimizer(model, lr, optimizer):
    if optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-6)
    return optimizer


def get_scheduler(optimizer, iterations, scheduler):
    if scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, iterations)
    elif scheduler == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[iterations / 4, iterations / 2], gamma=0.5)
    return scheduler


class PGD_Attack(nn.Module):
    def __init__(self, model, device, epsilon, iterations, optimizer, scheduler):
        super(PGD_Attack, self).__init__()

        self.epsilon = epsilon / 255
        self.lr = self.epsilon / 4
        self.attack_steps = iterations
        self.model = model
        self.device = device
        self.optimizer = get_optimizer(self.model, self.lr, optimizer)
        self.scheduler = get_scheduler(self.optimizer, self.attack_steps, scheduler)

    def forward(self, clean_images, true_labels):

        manipulated_images = torch.clone(clean_images)

        loss_fn = nn.CrossEntropyLoss()

        for _ in range(self.attack_steps):
            manipulated_images.requires_grad = True

            self.optimizer.zero_grad()

            logits = self.model(manipulated_images).to(self.device)

            loss = loss_fn(logits, true_labels).to(self.device)

            loss.backward()

            self.optimizer.step()

            self.scheduler.step()

            grad = manipulated_images.grad

            grad = grad.sign()

            manipulated_images = manipulated_images + self.lr * grad

            perturbation = manipulated_images - clean_images

            perturbation = torch.clamp(perturbation, min=-self.epsilon, max=self.epsilon)

            manipulated_images = clean_images + perturbation

            manipulated_images = torch.clamp(manipulated_images, min=0, max=1)

            manipulated_images = manipulated_images.detach()

        return manipulated_images