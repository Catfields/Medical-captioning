import pandas as pd
import re
from collections import Counter
import pickle
import os

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
REPORTS_PATH = os.path.join(SCRIPT_DIR, 'data/indiana_reports.csv')
PROJECTIONS_PATH = os.path.join(SCRIPT_DIR, 'data/indiana_projections.csv')
IMAGE_BASE_PATH = os.path.join(SCRIPT_DIR, 'data/images/images_normalized/')

# --- 1. Load and Merge Data ---
def load_and_merge_data():
    """Loads and merges the report and projection data."""
    reports_df = pd.read_csv(REPORTS_PATH)
    projections_df = pd.read_csv(PROJECTIONS_PATH)

    # Merge based on 'uid'
    merged_df = pd.merge(projections_df, reports_df, on='uid')

    # We are interested in the frontal images for now
    frontal_df = merged_df[merged_df['projection'] == 'Frontal'].copy()

    # Create full image path
    frontal_df['image_path'] = IMAGE_BASE_PATH + frontal_df['filename']

    # Select and rename columns for clarity
    processed_df = frontal_df[['image_path', 'findings', 'impression']].copy()
    processed_df.dropna(subset=['impression'], inplace=True)

    return processed_df

# --- 2. Clean and Tokenize Text ---
def clean_text(text):
    """Cleans and tokenizes the report text."""
    # Lowercase
    text = text.lower()
    # Remove non-alphanumeric characters (keeping spaces)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Tokenize by splitting on whitespace
    tokens = text.split()
    return tokens

# --- 3. Build Vocabulary ---
class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            for word in sentence:
                frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text_tokens):
        return [self.stoi.get(word, self.stoi["<UNK>"]) for word in text_tokens]


from sklearn.model_selection import train_test_split

# --- Main Execution ---
if __name__ == '__main__':
    print("Starting data preprocessing...")

    # 1. Load and merge
    data_df = load_and_merge_data()
    print(f"Loaded and merged data. Found {len(data_df)} frontal images with reports.")

    # 2. Clean impressions
    data_df['tokens'] = data_df['impression'].apply(clean_text)
    print("Cleaned and tokenized impressions.")

    # 3. Split data into train, validation, and test sets
    train_df, test_df = train_test_split(data_df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42) # 0.2 * 0.5 = 0.1 for val and test

    print(f"Split data into {len(train_df)} training, {len(val_df)} validation, and {len(test_df)} test samples.")

    # 4. Build Vocabulary on training data only
    train_impressions = train_df['tokens'].tolist()
    vocab = Vocabulary(freq_threshold=5) # Only include words that appear at least 5 times
    vocab.build_vocabulary(train_impressions)
    print(f"Built vocabulary with {len(vocab)} words based on the training set.")

    # 5. Save processed data and vocabulary
    train_path = os.path.join(SCRIPT_DIR, 'data/train.csv')
    val_path = os.path.join(SCRIPT_DIR, 'data/val.csv')
    test_path = os.path.join(SCRIPT_DIR, 'data/test.csv')
    vocab_path = os.path.join(SCRIPT_DIR, 'data/vocab.pkl')

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

    print("Preprocessing complete!")
    print(f"Saved training data to '{train_path}'")
    print(f"Saved validation data to '{val_path}'")
    print(f"Saved test data to '{test_path}'")
    print(f"Saved vocabulary to '{vocab_path}'")