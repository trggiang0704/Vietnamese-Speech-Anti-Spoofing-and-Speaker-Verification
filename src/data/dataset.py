import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

class VSASVDataset(Dataset):
    def __init__(self, metadata_path, data_dir):
        """
        Args:
            metadata_path (str): Đường dẫn tới file csv (train_metadata.csv, ...)
            data_dir (str): Thư mục gốc chứa audio đã preprocessing (datasets/preprocessing)
        """
        self.data_dir = data_dir
        self.metadata = pd.read_csv(metadata_path)
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # 1. Tái tạo đường dẫn file dựa vào Phase 2 (split / utt_type / filename)
        # Vì filename trong csv có dạng "audio/bonafide/utt_xyz.wav", ta chỉ lấy basename
        basename = os.path.basename(str(row['filename']))
        if not basename.endswith('.wav'):
            basename += '.wav'
            
        audio_path = os.path.join(self.data_dir, row['split'], row['utt_type'], basename)
        
        # 2. Đọc audio bằng torchaudio (File đã được chuẩn hóa 16kHz, mono, 3s ở Phase 2)
        waveform, sr = torchaudio.load(audio_path)
        
        # 3. Phân loại nhãn (Bonafide = 1, Spoof = 0)
        # Spoof bao gồm: adversarial_attack, voice_conversion, replay
        label = 1 if row['utt_type'] == 'bonafide' else 0
        
        # Đưa label về dạng float tensor để dùng hàm loss BCE (Binary Cross Entropy)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        # Trả về thêm utt_type để phục vụ đánh giá per-spoof (OOD) sau này
        utt_type = row['utt_type']
        
        return waveform, label_tensor, utt_type

# Test thử xem code có lỗi không
if __name__ == "__main__":
    test_ds = VSASVDataset(
        metadata_path="datasets/preprocessing/val_metadata.csv", 
        data_dir="datasets/preprocessing"
    )
    print(f"[INFO] Dataset size: {len(test_ds)}")
    wav, lbl, u_type = test_ds[0]
    print(f"Sample 0 - Shape: {wav.shape}, Label: {lbl}, Type: {u_type}")