import numpy as np
import pandas as pd
from scipy.ndimage import rotate, shift
import struct

# Constants
IMAGE_SIZE = 28
ROTATION_ANGLE = 10
SHIFT_VAL = 2

def load_mnist(file_path):
    with open(file_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, 'Magic number mismatch'
        assert rows == cols == IMAGE_SIZE, 'Image dimensions mismatch'
        images = np.fromfile(f, dtype=np.uint8).reshape(num_images, IMAGE_SIZE, IMAGE_SIZE)

    return images

def transform_images(images):
    # Add small rotations
    images = rotate(images, np.random.uniform(-ROTATION_ANGLE, ROTATION_ANGLE), axes=(1, 2), reshape=False)
    
    # Add small shifts
    images = shift(images, (0, np.random.randint(-SHIFT_VAL, SHIFT_VAL), np.random.randint(-SHIFT_VAL, SHIFT_VAL)))
    
    # Clip to original range and convert back to original type
    images = np.clip(images, 0, 255).astype(np.uint8)

    return images

def save_to_csv(images, file_path):
    # Flatten images and convert to DataFrame
    df = pd.DataFrame(images.reshape(len(images), -1))

    # Save to CSV
    df.to_csv(file_path, index=False)

def main():
    images = load_mnist('./mnist/t10k-images-idx3-ubyte')
    images = transform_images(images)
    save_to_csv(images, './mnist/transformed_images.csv')

if __name__ == '__main__':
    main()

