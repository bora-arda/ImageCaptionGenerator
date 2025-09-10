from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.transform import resize
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
