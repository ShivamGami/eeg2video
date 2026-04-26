import os
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class EEGVideoDataset(Dataset):
    def __init__(self, data_dir, mode='train'):
        self.data_dir = data_dir
        
        # Vipresh's Fixed Labels Load karo
        label_path = "/home/teaching/TEAM_22_DATASET/processed/processed/dynamics_labels_fixed_BINARY.npy"
        if os.path.exists(label_path):
            self.all_labels = np.load(label_path)
        else:
            print(f"⚠️ WARNING: Label file not found at {label_path}")
            self.all_labels = None
        
        # 1. Sabhi split files ko merge karna (Full 291k)
        split_files = ["train_split.txt", "val_split.txt", "test_split.txt"]
        all_ids = []
        for s_file in split_files:
            path = os.path.join(data_dir, s_file)
            if os.path.exists(path):
                with open(path, 'r') as f:
                    all_ids.extend([line.strip() for line in f.readlines() if line.strip()])
        
        # 2. Proper 70/15/15 Split from Total Data (Logic preserved)
        train_ids, temp_ids = train_test_split(all_ids, test_size=0.30, random_state=42)
        val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=42)
        
        if mode == 'train':
            self.sample_ids = train_ids
        elif mode == 'val':
            self.sample_ids = val_ids
        else:
            self.sample_ids = test_ids
            
        print(f"📊 DATASET LOADED | Mode: {mode.upper()} | Samples: {len(self.sample_ids)}")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        s_id = self.sample_ids[idx]
        
        # File paths
        eeg_path = os.path.join(self.data_dir, f'eeg_sample_{s_id}.pt')
        text_path = os.path.join(self.data_dir, f'text_sample_{s_id}.pt')
        video_path = os.path.join(self.data_dir, f'video_sample_{s_id}.pt')
        
        # Loading tensors
        eeg = torch.load(eeg_path, map_location='cpu')
        text = torch.load(text_path, map_location='cpu')
        video = torch.load(video_path, map_location='cpu')
        
        # --- FIXED LOGIC START ---
        if self.all_labels is not None:
            label_idx = int(s_id)
            # Agar index 291060 hai, toh usse 291059 bana do (aakhri valid index)
            if label_idx >= len(self.all_labels):
                label_idx = len(self.all_labels) - 1
            
            label = torch.tensor(self.all_labels[label_idx], dtype=torch.float32)
        else:
            label = torch.tensor(0.0)
        # --- FIXED LOGIC END ---

        return eeg, text, video, label