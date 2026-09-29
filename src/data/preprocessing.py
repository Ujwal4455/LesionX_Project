from PIL import Image
import numpy as np

def load_image(path,size=384):
    return np.asarray(Image.open(path).convert('RGB').resize((size,size)),dtype=np.float32)/255.
