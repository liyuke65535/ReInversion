import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def seed_everything(seed: int = 42):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)
