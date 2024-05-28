from hezar.trainer import Trainer, TrainerConfig
from hezar.constants import TaskType
from hezar.preprocessors import Preprocessor, ImageProcessorConfig
from hezar.models import CRNNImage2Text, CRNNImage2TextConfig

from hezar.data import OCRDataset, OCRDatasetConfig

from PIL import Image
from pathlib import Path
import pandas as pd
from constant import id2label

dataset_config = OCRDatasetConfig(
    path='C:\\Users\\Faraz\\PycharmProjects\\File\\data\\OCR\\data\\processed',
    # path='data/processed',
    task=TaskType.IMAGE2TEXT,
    image_processor_config=ImageProcessorConfig(
        gray_scale=True,
        mean=[0.6595],
        mirror=True,
        rescale=0.00392156862745098,
        size=(256, 64),
        std=[.1501]
    ),
    max_length=48,
    reverse_digits=True,
    text_split_type='char_split',
    id2label=id2label
)


class BringerOfTheLight(OCRDataset):
    def __init__(self, config: OCRDatasetConfig, split=None, **kwarg):
        super(BringerOfTheLight, self).__init__(config, split, **kwarg)
        self.valid_indices = self._get_valid_indices()

    def _load(self, split=None):
        return pd.read_csv(Path(self.config.path).joinpath(split).joinpath('image_data.txt'))

    def _get_valid_indices(self):
        valid_indices = []
        n_skipped = 0
        for index in range(len(self.data)):
            try:
                item = self.data.iloc[index]
                filename = item.image_path
                text = item.text
                image = Image.open(Path(self.config.path).joinpath(self.split).joinpath(filename))
                image = image.convert("RGB")
                labels = self._text_to_tensor(text)
                valid_indices.append(index)
            except Exception as e:
                print(f"Skipping sample {index} due to error: {e}")
                n_skipped += 1

        print('Number of skipped samples: ', n_skipped)
        return valid_indices

    def __getitem__(self, index):
        index = self.valid_indices[index]
        item = self.data.iloc[index]
        filename = item.image_path
        text = item.text

        # Read image
        image = Image.open(Path(self.config.path).joinpath(self.split).joinpath(filename))
        image = image.convert("RGB")

        pixel_values = self.image_processor(image, return_tensors='pt')['pixel_values'][0]
        labels = self._text_to_tensor(text)

        inputs = {
            "pixel_values": pixel_values,
            "labels": labels,
        }
        return inputs

    def __len__(self):
        return len(self.valid_indices)


train_dataset = BringerOfTheLight(config=dataset_config, split='train')
eval_dataset = BringerOfTheLight(config=dataset_config, split='test')

dataset_path = 'hezarai/parsynth-ocr-200k'
base_model_path = 'hezarai/crnn-base-fa-64x256'

model = CRNNImage2Text(CRNNImage2TextConfig(
    id2label=train_dataset.config.id2label,
    map2seq_in_dim=2048,
    map2seq_out_dim=64))

preprocessor = Preprocessor.load(base_model_path)

train_config = TrainerConfig(
    output_dir='runs',
    task=TaskType.IMAGE2TEXT,
    device='cuda:0',
    init_weights_from=base_model_path,
    batch_size=16,
    num_epochs=200,
    metrics=["cer"],
    metric_for_best_model="cer",
    gradient_accumulation_steps=2,
    learning_rate=2e-4
)

trainer = Trainer(
    config=train_config,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    preprocessor=preprocessor,
    data_collator=train_dataset.data_collator
)

trainer.train()
