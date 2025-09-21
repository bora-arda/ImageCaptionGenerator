from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.transform import resize
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

class ImageFeatureExtractor:
    
    def __init__(self, target_size = (256, 256), hog_params = None):
       self.hog_params = hog_params or {
           'orientations': 9,
           'pixels_per_cell': (8, 8),
           'cells_per_block': (2, 2),
           'block_norm': 'L2-Hys',
           'visualize': False,
           'feature_vector': True
       }
       self.target_size = target_size

    def extract_hog_features(self, image_path: str, multi_channel = False):
        """ Extract HOG features from an image. """
        
        image = imread(str(image_path))
        if image.ndim == 3 and image.shape[-1] == 4 and not multi_channel:
            # RGBA → RGB → Gray
            image = image[..., :3]
            gray = rgb2gray(image)
        elif image.ndim == 3 and image.shape[-1] == 3 and not multi_channel:
            # RGB → Gray
            gray = rgb2gray(image)
        elif image.ndim == 2:  
            # Already grayscale
            gray = image
        else:
            raise ValueError(f"Unexpected image shape {image.shape}")
            
        gray = resize(gray, self.target_size, anti_aliasing=True)
        hog_vec = hog(gray, **self.hog_params)
            
        return hog_vec.astype(np.float32)


    def get_feature_dim(self):
        """Calculate HOG feature dimension"""
        h, w = self.target_size
        orientations = self.hog_params['orientations']
        pixels_per_cell = self.hog_params['pixels_per_cell']
        cells_per_block = self.hog_params['cells_per_block']
        
        cells_h = h // pixels_per_cell[0]
        cells_w = w // pixels_per_cell[1]
        blocks_h = cells_h - cells_per_block[0] + 1
        blocks_w = cells_w - cells_per_block[1] + 1
        
        return blocks_h * blocks_w * cells_per_block[0] * cells_per_block[1] * orientations








# def process_image_directory(input_dir, output_dir, stats_path = "data/hog_stats.json", params = None):
#     """
#     Walk input_dir, compute HOG for each image and save as .npy in output_dir.
#     Also compute simple stats (mean/std) saved to stats_path.

#     Parameters:
#     - input_dir: Directory containing input images.
#     - output_dir: Directory to save the extracted HOG features.
#     - stats_path: Path to save the statistics of the extracted features.
#     - params: Dictionary of parameters for HOG extraction.
#     """
#     input_dir = Path(input_dir)
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
#     params = params or {}
#     hog_files = []
#     all_means = []
#     all_stds = []
    
#     for p in tqdm(sorted(input_dir.iterdir())):
#         if p.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
#             continue
#         try:
#             vec = extract_hog_features(p, **params)
#         except Exception as e:
#             print(f"Skipping {p} due to error: {e}")
#             continue
        
#         np.save(output_dir / (p.stem + '.npy'), vec)
#         hog_files.append(p.name)
#         all_means.append(float(np.mean(vec)))
#         all_stds.append(float(np.std(vec)))
#     stats = {
#         'n_files': len(hog_files),
#         'mean_of_means': float(np.mean(all_means)) if all_means else 0.0,
#         'mean_of_stds': float(np.mean(all_stds)) if all_stds else 0.0,
#         'params': params
#     }
#     Path(stats_path).parent.mkdir(parents = True, exist_ok = True)
#     with open(stats_path, 'w') as f:
#         json.dump(stats, f, indent = 2)
#     return stats


# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input_dir', required = True)
#     parser.add_argument('--output_dir', required = True)
#     parser.add_argument('--stats_path', default = 'data/hog_stats.json')
#     parser.add_argument('--resize_h', type = int, default = 128)
#     parser.add_argument('--resize_w', type = int, default = 128)
#     args = parser.parse_args()
#     params = {'pixels_per_cell': (8, 8), 'cells_per_block': (2, 2), 'resize_to': (args.resize_h, args.resize_w)}
#     print(process_image_directory(args.input_dir, args.output_dir, args.stats_path, params))