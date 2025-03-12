import numpy as np
import os
import cv2
from concurrent.futures import ThreadPoolExecutor

def process_npy_to_image(npy_path, output_folder):
    # Load the .npy file
    cqt_data = np.load(npy_path)

    # Ensure values are within -120 to 0 dB (faster than np.clip)
    cqt_data[cqt_data < -120] = -120
    cqt_data[cqt_data > 0] = 0

    # Normalize to [0, 255] (in-place for efficiency)
    cqt_image = ((cqt_data + 120) * (255 / 120)).astype(np.uint8)

    # Preserve filename and save as .png using OpenCV
    filename = os.path.basename(npy_path).replace('.npy', '.png')
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, cqt_image)

    return filename

def convert_npy_to_images(input_folder, output_folder, num_workers=8):
    os.makedirs(output_folder, exist_ok=True)
    npy_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.npy')]

    print(f"Processing {len(npy_files)} files using {num_workers} threads...")

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(lambda f: process_npy_to_image(f, output_folder), npy_files))

    print(f"✅ Conversion complete. Saved {len(results)} images to {output_folder}")

# Example usage
input_folder = "D:/Code playground/seminar_audioTab_/cqt_images"
output_folder = "D:/Code playground/seminar_audioTab_/cqt_images"
convert_npy_to_images(input_folder, output_folder, num_workers=12)
