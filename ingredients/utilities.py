import random
import numpy as np
import torch


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