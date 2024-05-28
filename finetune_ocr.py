import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from hezar.data import Dataset, OCRDatasetConfig, OCRDataset
from hezar.models import CRNNImage2TextConfig, CRNNImage2Text
from hezar.preprocessors import ImageProcessor, ImageProcessorConfig
from hezar.trainer import Trainer, TrainerConfig


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


# Load dataset to calculate mean and std
data_path = f"C:\\Users\\Faraz\\PycharmProjects\\File\\data\\OCR\\processed\\v2\\image_data.csv"
data = pd.read_csv(data_path, encoding='utf-8')
base_path = f'C:\\Users\\Faraz\\PycharmProjects\\File\\data\\OCR\\processed\\v2'

image_paths = [os.path.join(base_path, str(path)) for path in data['image_path']]
mean, std = calculate_mean_std(image_paths)

print(f"Calculated mean: {mean}, std: {std}")

version = 'v2'
train_path = f'C:\\Users\\Faraz\\PycharmProjects\\hezar\\examples\\train'
log_name = f'crnn-fa-{version}'
log_file_name = f"{log_name}.{1}"

i = 1
flag = True

while flag:
    log_file_name = f"{log_name}.{i}"
    TEMP_PATH = os.path.join(train_path, log_file_name)

    if os.path.isdir(TEMP_PATH):
        i += 1
    else:
        flag = False
        os.mkdir(os.path.join(TEMP_PATH))


class PersianOCRDataset(OCRDataset):
    def __init__(self, config: OCRDatasetConfig, split=None, **kwargs):
        super().__init__(config=config, split=split, **kwargs)

    def _load(self, split=None):
        data = pd.read_csv(self.config.path, encoding='utf-8')
        return data

    def __getitem__(self, index):
        base_path = f'C:\\Users\\Faraz\\PycharmProjects\\File\\data\\OCR\\processed\\{version}'
        path = os.path.join(base_path, str(self.data.iloc[index]['image_path']))
        text = self.data.iloc[index]['text']

        pixel_values = self.image_processor(path, return_tensors="pt")["pixel_values"][0]
        labels = self._text_to_tensor(text)
        inputs = {
            "pixel_values": pixel_values,
            "labels": labels,
        }
        return inputs


dataset_config = OCRDatasetConfig(
    path=data_path,
    text_split_type="char_split",
    text_column="text",
    images_paths_column="image_path",
    reverse_digits=False,
    image_processor_config=ImageProcessorConfig(
        gray_scale=True,
        mean=[mean],
        std=[std],
        mirror=False,
        rescale=1 / 255.0,
        size=(256, 64),
    )
)

train_dataset = PersianOCRDataset(config=dataset_config, split="train")
eval_dataset = PersianOCRDataset(config=dataset_config, split="test")

model = CRNNImage2Text(
    CRNNImage2TextConfig(
        id2label=train_dataset.config.id2label,
        map2seq_in_dim=2048,
        map2seq_out_dim=96
    )
)
preprocessor = ImageProcessor(train_dataset.config.image_processor_config)

train_config = TrainerConfig(
    output_dir=log_file_name,
    task="image2text",
    init_weights_from="hezarai/crnn-fa-printed-96-long",
    device="cuda:0",
    seed=42,
    batch_size=1,
    num_epochs=20,
    learning_rate=2e-5,
    metrics=["cer"],
    metric_for_best_model="cer"
)

trainer = Trainer(
    config=train_config,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=train_dataset.data_collator,
    preprocessor=preprocessor,
)
trainer.train()
