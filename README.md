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