import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset


class ImageCaptionDataset(Dataset):
    def __init__(self, captions: str, image_dir: str, ):
        pass