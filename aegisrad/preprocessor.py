import numpy as np
from PIL import Image
from config import IMAGE_SIZE, PIXEL_MEAN, PIXEL_STD


class Preprocessor:
    def __init__(self):
        self.size = IMAGE_SIZE
        self.mean = np.array(PIXEL_MEAN, dtype=np.float32)
        self.std  = np.array(PIXEL_STD,  dtype=np.float32)

    def process(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((self.size, self.size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        arr = arr.transpose(2, 0, 1)        # HWC -> CHW
        arr = np.expand_dims(arr, axis=0)   # -> [1, 3, 224, 224]
        return arr