import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
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
                    'image_path': str(Path('../data/raw/train/train') / (str(row['image_id']) + '.jpg')),
                    'caption': str(row['caption'])
                })
        elif self.captions_dir.endswith('.json'):
            df = pd.read_json(self.captions_dir)
            for _, row in df.iterrows():
                data.append({
                    'image_path': str(Path('../data/raw/train/train') / (str(row['image_id']) + '.jpg')),
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
        
        tokenized_caption = self.tokenizer.encode(item['caption'])
        tokenized_caption = self.tokenizer.apply_bos_eos(tokenized_caption)
        tokenized_caption = tokenized_caption[:self.max_caption_length-1] + [self.tokenizer.sp.eos_id()]
        
        return {
            'image_features': image_hog_features,
            'tokenized_caption': tokenized_caption,
            'caption_length': len(tokenized_caption),
            'original_caption': item['caption']
        }
        
    def collate_fn(self, batch: list[dict]) -> dict:
        """Custom collate function for DataLoader"""
        
        image_features = [item['image_features'] for item in batch]
        tokenized_captions = [item['tokenized_caption'] for item in batch]
        caption_lengths = [item['caption_length'] for item in batch]
        original_captions = [item['original_caption'] for item in batch]
        
        image_features = torch.stack(image_features)
        
        tokenized_captions = pad_sequence(tokenized_captions, batch_first=True, padding_value=self.tokenizer.sp.pad_id())
        
        return {
        'image_features': image_features,
        'tokenized_captions': tokenized_captions,
        'caption_lengths': torch.tensor(caption_lengths, dtype=torch.long),
        'original_captions': original_captions
        }
        