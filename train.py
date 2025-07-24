import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pickle
import os

from model import EncoderCNN, DecoderRNN
from dataset import XRayDataset
from preprocess import Vocabulary # We need the Vocabulary class definition

# --- Hyperparameters and Configuration ---
# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV_PATH = os.path.join(SCRIPT_DIR, 'data/train.csv')
VAL_CSV_PATH = os.path.join(SCRIPT_DIR, 'data/val.csv')
VOCAB_PATH = os.path.join(SCRIPT_DIR, 'data/vocab.pkl')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'models/')

# Model parameters
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 1

# Training parameters
NUM_EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 0.001

# --- Main Training Function ---
def main():
    # Create model directory if it doesn't exist
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    # --- 1. Load Vocabulary and Data ---
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Custom collate_fn to handle padding
    def collate_fn(data):
        data.sort(key=lambda x: len(x[1]), reverse=True)
        images, captions = zip(*data)
        images = torch.stack(images, 0)
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]
        return images, targets, lengths

    # Build data loaders
    train_dataset = XRayDataset(data_csv_path=TRAIN_CSV_PATH, vocab=vocab, transform=transform)
    val_dataset = XRayDataset(data_csv_path=VAL_CSV_PATH, vocab=vocab, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # --- 2. Build the models ---
    encoder = EncoderCNN(EMBED_SIZE).to(DEVICE)
    decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, len(vocab), NUM_LAYERS).to(DEVICE)

    # --- 3. Loss and Optimizer ---
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.embed.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)

    # --- 4. Train the models ---
    total_step = len(train_loader)
    print(f"Starting training on {DEVICE}...")

    for epoch in range(NUM_EPOCHS):
        encoder.train()
        decoder.train()
        for i, (images, captions, lengths) in enumerate(train_loader):
            images = images.to(DEVICE)
            captions = captions.to(DEVICE)

            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features, captions)

            packed_outputs = nn.utils.rnn.pack_padded_sequence(outputs, lengths, batch_first=True)[0]
            targets = nn.utils.rnn.pack_padded_sequence(captions, lengths, batch_first=True)[0]

            loss = criterion(packed_outputs, targets)
            
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i}/{total_step}], Loss: {loss.item():.4f}, Perplexity: {torch.exp(loss).item():5.4f}')

        # --- 5. Validate the model ---
        encoder.eval()
        decoder.eval()
        val_loss = 0
        with torch.no_grad():
            for images, captions, lengths in val_loader:
                images = images.to(DEVICE)
                captions = captions.to(DEVICE)
                features = encoder(images)
                outputs = decoder(features, captions)
                
                packed_outputs = nn.utils.rnn.pack_padded_sequence(outputs, lengths, batch_first=True)[0]
                targets = nn.utils.rnn.pack_padded_sequence(captions, lengths, batch_first=True)[0]
                
                loss = criterion(packed_outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        print(f'\nEpoch [{epoch+1}/{NUM_EPOCHS}], Validation Loss: {val_loss:.4f}, Validation Perplexity: {torch.exp(torch.tensor(val_loss)):5.4f}\n')

        # Save the model checkpoints
        torch.save(decoder.state_dict(), os.path.join(MODEL_PATH, f'decoder-{epoch+1}.ckpt'))
        torch.save(encoder.state_dict(), os.path.join(MODEL_PATH, f'encoder-{epoch+1}.ckpt'))

    print("Training finished.")

if __name__ == '__main__':
    main()