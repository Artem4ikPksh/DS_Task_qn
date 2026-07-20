# inference.py
import os
import argparse
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import models
import torch.nn as nn

# ==========================================
# 1. MODEL ARCHITECTURE (MUST MATCH TRAIN.PY)
# ==========================================
class SiameseEmbeddingNet(nn.Module):
    def __init__(self):
        super(SiameseEmbeddingNet, self).__init__()
        backbone = models.resnet18(weights=None)  # Weights will be loaded from our file
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
    def forward_once(self, x):
        output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        output = F.normalize(output, p=2, dim=1)
        return output

    def forward(self, input_a, input_b):
        return self.forward_once(input_a), self.forward_once(input_b)

# ==========================================
# 2. IMAGE PREPROCESSING FUNCTION
# ==========================================
def preprocess_image(img_path, img_size=(256, 256)):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, img_size)
    
    # Tensor preparation for PyTorch (C, H, W)
    img_tensor = img_resized.astype(np.float32) / 255.0
    img_tensor = img_tensor.transpose(2, 0, 1)
    img_tensor = torch.tensor(img_tensor).unsqueeze(0)  # adding batch dimension [1, 3, H, W]
    return img_rgb, img_tensor

# ==========================================
# 3. KEYPOINT MATCHING AND VISUALIZATION
# ==========================================
def match_and_visualize(img_path_a, img_path_b, model, device):
    # Load and prepare images
    orig_a, tensor_a = preprocess_image(img_path_a)
    orig_b, tensor_b = preprocess_image(img_path_b)
    
    # Convert to grayscale for keypoint detection
    gray_a = cv2.cvtColor(orig_a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(orig_b, cv2.COLOR_RGB2GRAY)
    
    # Check image similarity via our model
    model.eval()
    with torch.no_grad():
        emb_a = model.forward_once(tensor_a.to(device))
        emb_b = model.forward_once(tensor_b.to(device))
        # Cosine similarity between images (1 - identical, 0 - completely different)
        similarity = torch.mm(emb_a, emb_b.t()).item()
    
    # Use SIFT to generate candidate keypoints
    sift = cv2.SIFT_create(nfeatures=1000)
    kpts_a, desc_a = sift.detectAndCompute(gray_a, None)
    kpts_b, desc_b = sift.detectAndCompute(gray_b, None)
    
    if desc_a is None or desc_b is None:
        print("⚠️ Could not find descriptors on one of the images.")
        return

    # Find matches via BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(desc_a, desc_b, k=2)
    
    # Filtering using Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            
    # Additional geometric validation via RANSAC
    if len(good_matches) > 4:
        src_pts = np.float32([kpts_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kpts_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
    else:
        matches_mask = None
        print("⚠️ Not enough points for RANSAC geometric filtering.")

    # Convert back to BGR for OpenCV drawMatches rendering
    bgr_a = cv2.cvtColor(orig_a, cv2.COLOR_RGB2BGR)
    bgr_b = cv2.cvtColor(orig_b, cv2.COLOR_RGB2BGR)
    
    draw_params = dict(matchColor=(0, 255, 0),       # green lines for valid matches
                       singlePointColor=(255, 0, 0),  # red points for orphan keypoints
                       matchesMask=matches_mask,      # display only RANSAC verified points
                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    result_img = cv2.drawMatches(bgr_a, kpts_a, bgr_b, kpts_b, good_matches, None, **draw_params)
    
    # Save the result to disk
    output_path = "matching_result.png"
    cv2.imwrite(output_path, result_img)
    print(f"💾 Matching visualization saved to: {output_path}")
    
    # Plot the figure
    plt.figure(figsize=(15, 8))
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Satellite Matching (Model Similarity Score: {similarity:.4f})\nGreen lines show geometrically stable landmarks")
    plt.axis('off')
    plt.show()

# ==========================================
# 4. MAIN EXECUTION FUNCTION
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Inference Script for Image Matching")
    parser.add_argument("--csv_file", type=str, default="sentinel_dataset_index.csv")
    parser.add_argument("--weights", type=str, default="model_weights.pth")
    parser.add_argument("--pair_idx", type=int, default=0, help="Index of the CSV pair to verify")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Inference running on device: {device}")
    
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"❌ Weights file {args.weights} not found! Please run train.py first.")
        
    if not os.path.exists(args.csv_file):
        raise FileNotFoundError(f"❌ Index CSV file not found!")

    # Model initialization and loading trained weights
    model = SiameseEmbeddingNet().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    print("✅ Model weights successfully loaded.")
    
    # Select a pair for demonstration
    df = pd.read_csv(args.csv_file)
    if args.pair_idx >= len(df):
        print(f"⚠️ Index {args.pair_idx} is too large, selecting the first pair (idx=0).")
        args.pair_idx = 0
        
    row = df.iloc[args.pair_idx]
    print(f"🔍 Testing pair ID: {row['core_id']}")
    print(f"  Image A (s1): {row['image_a']}")
    print(f"  Image B (s2): {row['image_b']}")
    
    match_and_visualize(row['image_a'], row['image_b'], model, device)

if __name__ == "__main__":
    main()