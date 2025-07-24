import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class XRayDataset(Dataset):
    """Custom PyTorch Dataset for X-Ray images and their reports."""
    def __init__(self, data_csv_path, vocab, transform=None):
        """
        Args:
            data_csv_path (string): Path to the csv file with image paths and captions.
            vocab (Vocabulary): Vocabulary object.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = pd.read_csv(data_csv_path)
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_filename = os.path.basename(self.data.iloc[idx, 0])
        # Construct the correct path relative to the script's location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(script_dir, 'data', 'images', img_filename)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            # Fallback for cases where images might be in a nested 'images_normalized' folder
            img_path = os.path.join(script_dir, 'data', 'images', 'images_normalized', img_filename)
            image = Image.open(img_path).convert("RGB")

        caption = self.data.iloc[idx, 2] # 'impression' column
        tokens = str(caption).lower().split()

        # Numericalize the caption text
        caption_vec = []
        caption_vec.append(self.vocab.stoi["<START>"])
        caption_vec.extend(self.vocab.numericalize(tokens))
        caption_vec.append(self.vocab.stoi["<END>"])

        caption_tensor = torch.LongTensor(caption_vec)

        if self.transform:
            image = self.transform(image)

        return image, caption_tensor