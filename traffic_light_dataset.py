import os
import pandas as pd
from PIL import Image
from typing import Callable, Optional
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class TrafficLightDataset(Dataset):
    """LISA Traffic Light Dataset wrapper.

    Expects an annotations CSV containing at least columns: `Filename`, `Annotation tag`.
    The CSV is typically found in the `Annotations` directory of the dataset.

    Args:
        root: Root directory of the dataset (contains image folders and `Annotations/`).
       th to annotations CSV. If None, will look for one in annotation_csv: Pa `root/Annotations`.
        transform: Optional transform applied to PIL Image.
        target_transform: Optional transform applied to class index.
        class_map: Optional dict mapping raw annotation tags to desired class names.

    Default class_map reduces various tags to three states: red, yellow, green.
    """

    DEFAULT_CLASS_MAP = {
        'go': 'green', 'stop': 'red', 'warning': 'yellow', 'goForward': 'green',
        'goLeft': 'green', 'goRight': 'green', 'warningLeft': 'yellow', 'warningRight': 'yellow',
        'stopLeft': 'red', 'stopRight': 'red'
    }

    def __init__(self,
                 root: str,
                 annotation_csv: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 class_map: Optional[dict] = None):
        self.root = root
        if annotation_csv is None:
            # Try common filename
            candidate = os.path.join(root, 'Annotations', 'Annotations', 'daySequence1', 'frameAnnotationsBOX.csv')
            if not os.path.isfile(candidate):
                raise FileNotFoundError("Annotations CSV not found. Provide path explicitly.")
            annotation_csv = candidate
        self.annotation_csv = annotation_csv
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.target_transform = target_transform
        self.class_map = class_map or self.DEFAULT_CLASS_MAP

        df = pd.read_csv(self.annotation_csv, delimiter=';')
        # Normalize column names
        df.columns = [c.strip() for c in df.columns]
        if 'Filename' not in df.columns or 'Annotation tag' not in df.columns:
            raise ValueError("Annotation CSV must contain 'Filename' and 'Annotation tag' columns.")

        # Map tags
        df['MappedTag'] = df['Annotation tag'].map(lambda t: self.class_map.get(t, 'other'))
        # Filter unknown tags except keep only defined classes
        df = df[df['MappedTag'].isin(['red', 'yellow', 'green'])]
        # print(df.head())

        self.samples = df[['Filename', 'MappedTag']].values.tolist()
        self.classes = ['red', 'yellow', 'green']
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        rel_path, tag = self.samples[idx]
        img_path = os.path.join(self.root, rel_path)
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            img = self.transform(img)
        label = self.class_to_idx[tag]
        if self.target_transform:
            label = self.target_transform(label)
        return img, label

    def __repr__(self) -> str:
        return super().__repr__() + f"\nNumber of samples: {len(self)}\nClasses: {self.classes}"

if __name__ == "__main__":
    dataset = TrafficLightDataset(root='./lisa-traffic-light-dataset', annotation_csv='./lisa-traffic-light-dataset/Annotations/Annotations/daySequence1/frameAnnotationsBOX.csv', class_map={
        'go': 'green', 'stop': 'red', 'warning': 'yellow'
    })
    # print(f"Dataset size: {len(dataset)}")
    print(dataset)
    img, label = dataset[0]
    # print(f"Image: {img.shape} --> Label: {label}")