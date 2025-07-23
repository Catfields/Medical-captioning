import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    """Encoder model - uses a pretrained CNN to extract features."""
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # Use a pretrained ResNet-50, but remove the final fully connected layer
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False) # Freeze all layers

        modules = list(resnet.children())[:-1] # Remove the last fc layer
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        # Extract the features from the pretrained model
        features = self.resnet(images)
        # Reshape the features to be (batch_size, -1)
        features = features.view(features.size(0), -1)
        # Pass the features through the linear layer
        features = self.embed(features)
        # Pass through batch normalization
        features = self.bn(features)
        return features


class DecoderRNN(nn.Module):
    """Decoder model - uses an LSTM to generate captions."""
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        # Remove the <END> token from captions for training
        captions = captions[:, :-1]
        # Embed the captions
        embeddings = self.embed(captions)
        # Concatenate the image features and caption embeddings
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        # Pass through the LSTM
        hiddens, _ = self.lstm(inputs)
        # Pass through the linear layer
        outputs = self.linear(hiddens)
        return outputs

    def sample(self, features, states=None, max_len=20):
        """Greedy search to generate a caption for an image."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)              # sampled_ids: (batch_size, max_len)
        return sampled_ids