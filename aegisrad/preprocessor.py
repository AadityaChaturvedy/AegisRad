import numpy as np
from PIL import Image
from config import IMAGE_SIZE, PIXEL_MEAN, PIXEL_STD

class Preprocessor:
    def __init__(self):
        self.size = IMAGE_SIZE
        self.mean = np.array(PIXEL_MEAN, dtype=np.float32)
        self.std  = np.array(PIXEL_STD,  dtype=np.float32)

    def process(self, image_path: str) -> np.ndarray:
        # Load and immediately resize to avoid keeping large image in memory
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((self.size, self.size), Image.BILINEAR)  # In-place resize
        img = img.resize((self.size, self.size), Image.BILINEAR)
        
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        arr = arr.transpose(2, 0, 1)
        arr = np.expand_dims(arr, axis=0)
        
        # Free memory immediately
        del img
        
        return arr
