```python
# data_preprocessing.py

import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class TextImageDataset(Dataset):
    def __init__(self, text_image_pairs, transform=None):
        self.text_image_pairs = text_image_pairs
        self.transform = transform

    def __len__(self):
        return len(self.text_image_pairs)

    def __getitem__(self, idx):
        text, image_path = self.text_image_pairs[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {'text': text, 'image': image}

def get_dataloader(text_image_pairs, batch_size=4, image_size=(256, 256)):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    dataset = TextImageDataset(text_image_pairs, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
```
