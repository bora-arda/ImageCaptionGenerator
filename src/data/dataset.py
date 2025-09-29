import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
from src.tokenizer.tokenizer import BPETokenizer
from src.data.hog_extractor import ImageFeatureExtractor
from tqdm import tqdm
from datetime import datetime


class ImageCaptionDataset(Dataset):
    """Dataset Class for Image-Caption Pairs"""
    
    def __init__(self, captions_dir: str, image_dir: str, tokenizer: BPETokenizer, image_feature_extractor: ImageFeatureExtractor, max_caption_length: int = 250, hog_stats = None):
        self.captions_dir = captions_dir
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.feature_extractor = image_feature_extractor
        self.max_caption_length = max_caption_length
        self.hog_stats = hog_stats
        
        self.data = self._load_data()
        
    def _load_data(self) -> list[dict]:
        data = []
        
        if self.captions_dir.endswith(".csv"):
            df = pd.read_csv(self.captions_dir)
            for _, row in df.iterrows():
                data.append({
                    'image_path': str(self.image_dir / (row['image_id'] + '.jpg')),
                    'caption': str(row['caption'])
                })
        elif self.captions_dir.endswith('.json'):
            df = pd.read_json(self.captions_dir)
            for _, row in df.iterrows():
                data.append({
                    'image_path': str(self.image_dir / (row['image_id'] + '.jpg')),
                    'caption': str(row['caption'])
                })
                
        else:
            raise ValueError(f"Unsupported file type: {self.captions_dir}. Only .csv and .json are allowed.")
        
        return data
    
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int):
        """Get a single sample"""
        
        item = self.data[idx]
        
        image_hog_features = self.feature_extractor.extract_hog_features(item['image_path'])
        
        caption_tokens = self.tokenizer.encode(item['caption'])
    