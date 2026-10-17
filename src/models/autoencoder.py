import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class FlightWindowDataset(Dataset):
    """
    Creates sliding temporal windows from multivariate flight telemetry.
    """
    def __init__(self, data: np.ndarray, seq_len: int = 50):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        return self.data[idx:idx + self.seq_len]

class LSTMAutoencoder(nn.Module):
    """
    Encoder-Decoder architecture using LSTMs for time-series reconstruction.
    """
    def __init__(self, num_features: int, hidden_dim: int = 64, latent_dim: int = 32):
        super(LSTMAutoencoder, self).__init__()
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=num_features, 
            hidden_size=hidden_dim, 
            num_layers=1, 
            batch_first=True
        )
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=num_features, 
            num_layers=1, 
            batch_first=True
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Encode
        _, (hidden, _) = self.encoder_lstm(x)
        latent = torch.relu(self.encoder_fc(hidden.squeeze(0)))
        
        # Prepare Decoder input (repeat latent vector across sequence length)
        dec_input = torch.relu(self.decoder_fc(latent)).unsqueeze(1).repeat(1, seq_len, 1)
        
        # Decode
        decoded, _ = self.decoder_lstm(dec_input)
        return decoded

def train_autoencoder(parquet_path: str, epochs: int = 5, batch_size: int = 32, lr: float = 1e-3):
    """
    Loads an aligned flight parquet file and trains the LSTM Autoencoder.
    """
    if not os.path.exists(parquet_path):
        print(f"Parquet file not found at: {parquet_path}")
        return
        
    print(f"Loading flight data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # Drop non-numeric columns like timestamps if present
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=['timestamp'], errors='ignore')
    
    # Normalize features (Zero mean, unit variance)
    data_np = numeric_df.values
    data_mean = data_np.mean(axis=0)
    data_std = data_np.std(axis=0) + 1e-8
    data_normalized = (data_np - data_mean) / data_std
    
    # Setup Dataset and DataLoader
    seq_len = 50
    dataset = FlightWindowDataset(data_normalized, seq_len=seq_len)
    if len(dataset) == 0:
        print("Flight sequence is too short for the chosen window length.")
        return
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model, loss, and optimizer
    num_features = data_normalized.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = LSTMAutoencoder(num_features=num_features).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"Training LSTM Autoencoder on {device} for {epochs} epochs...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
            
        total_loss = epoch_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.6f}")
        
    # Save trained model weights
    os.makedirs("models", exist_ok=True)
    model_path = "models/lstm_autoencoder.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model successfully saved to {model_path}")

if __name__ == "__main__":
    import glob
    aligned_files = glob.glob("data/aligned/*.parquet")
    if aligned_files:
        train_autoencoder(aligned_files[0], epochs=3)
    else:
        print("No aligned parquet files found in data/aligned/. Please run align_channels.py first.")