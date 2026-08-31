import torch
import torch.nn as nn
from config import Config

class ConvSequenceAutoencoder(nn.Module):
    """
    A 1D-CNN based Autoencoder.
    We use CNN instead of native LSTM because standard PyTorch nn.LSTM
    is often incompatible with Differential Privacy (Opacus) without complex 
    workarounds. 1D-CNN captures temporal patterns (Conv1d over time axis) 
    while remaining 100% compliant with Opacus PrivacyEngine.
    """
    def __init__(self, seq_len=None, num_features=None):
        seq_len = Config.SEQ_LENGTH if seq_len is None else seq_len
        num_features = Config.NUM_FEATURES if num_features is None else num_features
        super(ConvSequenceAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        
        # Encoder: (Batch, Features, Seq_Len)
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=num_features, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # x input shape: (Batch, Seq_Len, Features)
        # Permute for Conv1d: (Batch, Features, Seq_Len)
        x = x.permute(0, 2, 1)
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        # Permute back to: (Batch, Seq_Len, Features)
        return decoded.permute(0, 2, 1)

def get_model():
    return ConvSequenceAutoencoder()
