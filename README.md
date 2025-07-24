# Medical Image Captioning

This project implements a deep learning model to automatically generate medical reports (captions) for chest X-ray images. It uses an Encoder-Decoder architecture with a pretrained CNN for feature extraction and an LSTM for text generation.

## Project Structure

```
.
├── data
│   ├── images
│   │   └── images_normalized
│   ├── indiana_projections.csv
│   ├── indiana_reports.csv
│   ├── processed_reports.csv
│   └── vocab.pkl
├── models
│   ├── encoder-5.ckpt
│   └── decoder-5.ckpt
├── preprocess.py
├── dataset.py
├── model.py
├── train.py
└── sample.py
```

- `data/`: Contains the dataset files. `images/` holds the X-ray images, while the `.csv` files contain the report data.
- `models/`: Stores the trained model checkpoints.
- `preprocess.py`: Script for cleaning and preparing the text data and creating a vocabulary.
- `dataset.py`: Defines the PyTorch `Dataset` for loading images and captions.
- `model.py`: Contains the `EncoderCNN` and `DecoderRNN` model definitions.
- `train.py`: The main script for training the model.
- `sample.py`: Script to generate a caption for a given image using the trained model.

## How to Run

### 1. Prerequisites

Make sure you have Python 3 and the following libraries installed:

- `torch`
- `torchvision`
- `pandas`
- `matplotlib`
- `Pillow`

You can install them using pip:

```bash
pip install torch torchvision pandas matplotlib Pillow
```

### 2. Data Preprocessing

First, run the preprocessing script to prepare the data:

```bash
python preprocess.py
```

This will create `processed_reports.csv` and `vocab.pkl` in the `data/` directory.

### 3. Train the Model

To train the model, run the training script:

```bash
python train.py
```

The trained model weights will be saved in the `models/` directory.

### 4. Generate a Sample Report

To generate a report for a sample image, use the `sample.py` script:

```bash
python sample.py --image_path /path/to/your/image.png
```

By default, it uses an image from the dataset. You can modify the path to test with your own images.

## Training Details

The model was trained with the following hyperparameters:

- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Batch Size**: 16
- **Number of Epochs**: 5
- **Embedding Size**: 256
- **Hidden LSTM Size**: 512
- **LSTM Layers**: 1

The dataset is split into training and validation sets, which are loaded from `data/train.csv` and `data/val.csv` respectively.

## Evaluation Results

The model was evaluated on the test set, and the following scores were obtained:

| Metric  | Score  |
|---------|--------|
| Bleu_1  | 0.3373 |
| Bleu_2  | 0.2386 |
| Bleu_3  | 0.1679 |
| Bleu_4  | 0.1228 |
| METEOR  | 0.3438 |
| ROUGE_L | 0.4116 |
| CIDEr   | 0.0781 |