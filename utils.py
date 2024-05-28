from __future__ import annotations

import csv

import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from pathlib import Path


# Calculate mean and std for the dataset
def calculate_mean_std(image_paths):
    means = []
    stds = []

    for img_path in tqdm(image_paths):
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img_array = np.array(img) / 255.0  # Normalize to [0, 1]

        means.append(np.mean(img_array))
        stds.append(np.std(img_array))

    mean = float(np.mean(means))  # Convert to Python float
    std = float(np.mean(stds))  # Convert to Python float
    return mean, std


def gpu_check():
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available!")
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")

        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
            print(f"Memory Cached: {torch.cuda.memory_reserved(i) / 1024 ** 2:.2f} MB")
    else:
        print("CUDA is not available.")


def combine_txt_files(list_paths: list, save_path: str | Path):
    # Read the input files into pandas DataFrames
    df1 = pd.read_csv(list_paths[0], encoding='utf-8')
    df2 = pd.read_csv(list_paths[1], encoding='utf-8')

    # Concatenate the DataFrames
    aggregated_df = pd.concat([df1, df2], ignore_index=True)

    # Save the aggregated DataFrame to a new text file
    aggregated_df.to_csv(save_path, index=False, encoding='utf-8')


if __name__ == "__main__":
    gpu_check()
    # combine_txt_files(['C:\\Users\\Faraz\\PycharmProjects\\File\\data\\OCR\\data\\processed\\train\\image_data_v2.txt',
    #                    'C:\\Users\\Faraz\\PycharmProjects\\File\\data\\OCR\\data\\processed\\train\\image_data_parsynth.txt'],
    #                   save_path='C:\\Users\\Faraz\\PycharmProjects\\File\\data\\OCR\\data\\processed\\train\\image_data.txt')
