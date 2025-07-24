import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pickle
import os
import argparse
from tqdm import tqdm

from model import EncoderCNN, DecoderRNN
from dataset import XRayDataset
from preprocess import Vocabulary
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

# --- Evaluation Function ---
def evaluate(encoder, decoder, data_loader, vocab, device):
    encoder.eval()
    decoder.eval()

    gts = {}
    res = {}
    img_id = 0

    with torch.no_grad():
        for images, captions, lengths in tqdm(data_loader):
            images = images.to(device)
            features = encoder(images)
            sampled_ids = decoder.sample(features)
            sampled_ids = sampled_ids[0].cpu().numpy()

            # Convert word_ids to words
            sampled_caption = []
            for word_id in sampled_ids:
                word = vocab.itos[word_id]
                if word == '<end>':
                    break
                if word != '<start>':
                    sampled_caption.append(word)
            
            sentence = ' '.join(sampled_caption)
            res[img_id] = [sentence]

            # Ground truth captions
            # For simplicity, we take the first caption as reference
            # In a real scenario, you might have multiple references
            ground_truth_caption = []
            for word_id in captions[0].cpu().numpy():
                word = vocab.itos[word_id]
                if word == '<end>':
                    break
                if word not in ['<start>', '<pad>']:
                    ground_truth_caption.append(word)
            gts[img_id] = [' '.join(ground_truth_caption)]
            img_id += 1

    return gts, res

# --- Main Execution --- 
def main(args):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load Vocabulary and Data ---
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Build data loader for the test set
    # Assuming you have a separate CSV for testing or are using a subset
    dataset = XRayDataset(data_csv_path=args.data_csv_path, vocab=vocab, transform=transform)
    
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

    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # --- Build the models ---
    encoder = EncoderCNN(args.embed_size).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)

    # Load the trained model weights
    encoder.load_state_dict(torch.load(args.encoder_path, map_location=device))
    decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))

    # --- Evaluate the Model ---
    print("Evaluating the model...")
    gts, res = evaluate(encoder, decoder, data_loader, vocab, device)

    # --- Calculate Scores ---
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(gts, res)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score

    print("Evaluation Scores:")
    for key, value in final_scores.items():
        print(f"{key}: {value:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_path', type=str, default='models/encoder-5.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/decoder-5.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--data_csv_path', type=str, default='data/test.csv', help='path for processed reports CSV')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    
    args = parser.parse_args()
    main(args)