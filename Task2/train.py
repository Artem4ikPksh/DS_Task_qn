# train.py
import os
import argparse
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models

# ==========================================
# 1. CONFIGURATION AND ARGUMENTS
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Training Pipeline for Sentinel-2 Image Matching")
    parser.add_argument("--csv_file", type=str, default="sentinel_dataset_index.csv", help="Path to data index CSV")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (small default for CPU stability)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output_weights", type=str, default="model_weights.pth", help="Path to save model weights")
    return parser.parse_args()

# ==========================================
# 2. DATASET FOR SIAMESE NETWORK
# ==========================================
class SentinelSiameseDataset(Dataset):
    def __init__(self, csv_file, img_size=(256, 256)):
        self.df = pd.read_csv(csv_file)
        self.img_size = img_size
        
    def __len__(self):
        return len(self.df)
    
    def _preprocess(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            # Backup empty tensor in case of file reading failure
            return torch.zeros((3, self.img_size[0], self.img_size[1]), dtype=torch.float32)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        
        # Image normalization [0, 1] and conversion to PyTorch format (C, H, W)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        return torch.tensor(img, dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Positive pair (same location in summer and winter seasons)
        img_summer = self._preprocess(row['image_a'])
        img_winter = self._preprocess(row['image_b'])
        
        # Negative pair (random winter image from a completely different location)
        if len(self.df) > 1:
            random_idx = (idx + np.random.randint(1, len(self.df))) % len(self.df)
        else:
            random_idx = idx
        img_negative = self._preprocess(self.df.iloc[random_idx]['image_b'])
        
        return img_summer, img_winter, img_negative

# ==========================================
# 3. MODEL ARCHITECTURE (SIAMESE RESNET)
# ==========================================
class SiameseEmbeddingNet(nn.Module):
    def __init__(self):
        super(SiameseEmbeddingNet, self).__init__()
        # Using pretrained ResNet18 as a baseline Feature Extractor
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Remove the final classification layer
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        
        # Projection head to obtain a compact feature vector (Embedding)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
    def forward_once(self, x):
        output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        # CRITICAL: L2-normalization constraints vectors onto a unit sphere, preventing 0.0000 loss
        output = F.normalize(output, p=2, dim=1)
        return output

    def forward(self, input_summer, input_winter, input_negative=None):
        output_summer = self.forward_once(input_summer)
        output_winter = self.forward_once(input_winter)
        
        if input_negative is not None:
            output_negative = self.forward_once(input_negative)
            return output_summer, output_winter, output_negative
            
        return output_summer, output_winter

# ==========================================
# 4. LOSS FUNCTION (ROBUST TRIPLET LOSS)
# ==========================================
class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        # For normalized spheres, the optimal margin lies in the 0.2 - 0.5 range
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        # Computation of classical Euclidean distance between feature vectors
        distance_positive = torch.pairwise_distance(anchor, positive, p=2)
        distance_negative = torch.pairwise_distance(anchor, negative, p=2)
        
        # Loss calculation with a threshold cutoff at zero using ReLU
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

# ==========================================
# 5. MAIN TRAINING LOOP
# ==========================================
def main():
    args = parse_args()
    
    # Device selection (defaults to cpu if old drivers are present)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training started on device: {device}")
    
    # Check for dataset index file existence
    if not os.path.exists(args.csv_file):
        raise FileNotFoundError(f"❌ Index file not found: {args.csv_file}. Please generate it in the notebook first.")
        
    # Initialize dataset
    dataset = SentinelSiameseDataset(csv_file=args.csv_file)
    
    # Protection against ZeroDivisionError and empty dataset
    current_batch_size = min(args.batch_size, len(dataset))
    if current_batch_size == 0:
        raise ValueError("❌ CSV file found, but it contains no records. Please check the parser!")
        
    print(f"📊 Total pairs in dataset: {len(dataset)}. Using batch size: {current_batch_size}")
    
    # Create DataLoader with drop_last=False
    train_loader = DataLoader(
        dataset, 
        batch_size=current_batch_size, 
        shuffle=True, 
        drop_last=False
    )
    
    # Initialize model, optimizer, and loss criterion
    model = SiameseEmbeddingNet().to(device)
    criterion = TripletLoss(margin=0.3)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print("📋 Beginning training epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (img_sum, img_win, img_neg) in enumerate(train_loader):
            img_sum = img_sum.to(device)
            img_win = img_win.to(device)
            img_neg = img_neg.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass through the Siamese network
            emb_sum, emb_win, emb_neg = model(img_sum, img_win, img_neg)
            
            # Loss computation
            loss = criterion(emb_sum, emb_win, emb_neg)
            
            # Backward propagation of error
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] -> Average Triplet Loss: {avg_loss:.6f}")
        
    # Saving the final weights for the task (.pth file)
    torch.save(model.state_dict(), args.output_weights)
    print(f"💾 Model weights successfully saved to file: {args.output_weights}")

if __name__ == "__main__":
    main()