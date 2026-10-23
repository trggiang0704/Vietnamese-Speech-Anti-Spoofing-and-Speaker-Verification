import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import soundfile as sf
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# Tải ECAPA-TDNN
from speechbrain.inference.speaker import EncoderClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# ==========================================
# 1. DATASET CHO TRAINING (SPEAKER CLASSIFICATION)
# ==========================================
class SpeakerDataset(Dataset):
    def __init__(self, metadata_path, data_dir, label_encoder):
        print(f"[INFO] Đọc dữ liệu Train từ: {metadata_path}")
        df = pd.read_csv(metadata_path)
        # CHỈ LẤY BONAFIDE
        self.df = df[df['utt_type'] == 'bonafide'].copy()
        
        self.data_dir = data_dir
        self.label_encoder = label_encoder
        
        # Mã hóa tên speaker thành số nguyên (0 đến N-1)
        self.df['label_idx'] = self.label_encoder.transform(self.df['speaker'])
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = os.path.basename(str(row['filename']).replace('\\', '/'))
        filepath = os.path.join(self.data_dir, 'train', 'bonafide', filename)
        
        wav, _ = sf.read(filepath)
        sig = torch.tensor(wav, dtype=torch.float32)
        label = torch.tensor(row['label_idx'], dtype=torch.long)
        return sig, label

# ==========================================
# 2. MÔ HÌNH WRAPPER (THÊM HEAD PHÂN LOẠI)
# ==========================================
class ECAPAFineTuner(nn.Module):
    def __init__(self, num_classes, freeze_backbone=True, device='cuda:0'):
        super().__init__()
        print(f"[INFO] Tải SpeechBrain ECAPA-TDNN...")
        self.ecapa = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            savedir="checkpoints/pretrained_ecapa",
            run_opts={"device": device}
        )
        
        # Đóng băng hoặc Mở khóa Backbone
        if freeze_backbone:
            print("[INFO] Chế độ: FREEZE BACKBONE (Chỉ train Head)")
            for param in self.ecapa.parameters():
                param.requires_grad = False
        else:
            print("[INFO] Chế độ: UNFREEZE BACKBONE (Fine-tune toàn bộ)")
            for param in self.ecapa.parameters():
                param.requires_grad = True
                
        # Lớp Head phân loại danh tính người nói (192-dim -> num_classes)
        self.head = nn.Linear(192, num_classes)
        
    def forward(self, x):
        # Ép qua các module tính feature của SpeechBrain
        feats = self.ecapa.mods.compute_features(x)
        feats = self.ecapa.mods.mean_var_norm(feats, torch.ones(x.shape[0], device=x.device))
        embeddings = self.ecapa.mods.embedding_model(feats).squeeze(2).squeeze(1) # Shape: (Batch, 192)
        
        logits = self.head(embeddings)
        return logits, embeddings

# ==========================================
# 3. HÀM TÍNH EER CHO VALIDATION
# ==========================================
def calculate_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

def validate_eer(model, val_pairs_path, data_dir, device):
    model.eval()
    df = pd.read_csv(val_pairs_path)
    cos_sim = nn.CosineSimilarity(dim=0)
    
    y_true = []
    y_scores = []
    
    print("\n[INFO] Đang chạy Validation EER trên 5000 cặp Val...")
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Val EER", leave=False):
            f1 = os.path.join(data_dir, 'val', 'bonafide', os.path.basename(str(row['audio1']).replace('\\', '/')))
            f2 = os.path.join(data_dir, 'val', 'bonafide', os.path.basename(str(row['audio2']).replace('\\', '/')))
            
            wav1, _ = sf.read(f1)
            wav2, _ = sf.read(f2)
            
            sig1 = torch.tensor(wav1, dtype=torch.float32).unsqueeze(0).to(device)
            sig2 = torch.tensor(wav2, dtype=torch.float32).unsqueeze(0).to(device)
            
            _, emb1 = model(sig1)
            _, emb2 = model(sig2)
            
            score = cos_sim(emb1.squeeze(), emb2.squeeze()).item()
            y_scores.append(score)
            y_true.append(row['label'])
            
    eer = calculate_eer(y_true, y_scores)
    return eer

# ==========================================
# 4. VÒNG LẶP HUẤN LUYỆN CHÍNH
# ==========================================
def train(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Chuẩn bị tên thư mục lưu kết quả
    run_name = "ecapa_stage_A_freeze" if args.stage == 'A' else "ecapa_stage_B_finetune"
    save_dir = f"checkpoints/{run_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Xây dựng bộ từ điển nhãn (Label Encoder) cho toàn bộ speaker trong Train
    train_meta_path = "datasets/preprocessing/train_metadata.csv"
    train_df = pd.read_csv(train_meta_path)
    train_df = train_df[train_df['utt_type'] == 'bonafide']
    
    le = LabelEncoder()
    le.fit(train_df['speaker'])
    num_classes = len(le.classes_)
    print(f"[INFO] Tổng số danh tính người nói (Classes): {num_classes}")

    # 2. Khởi tạo Data & DataLoader
    train_dataset = SpeakerDataset(train_meta_path, "datasets/preprocessing", le)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    # 3. Khởi tạo Model
    freeze = True if args.stage == 'A' else False
    # CHÚ Ý: Đã ép kiểu str(device) để SpeechBrain không bị lỗi TypeError
    model = ECAPAFineTuner(num_classes=num_classes, freeze_backbone=freeze, device=str(device)).to(device)
    
    # Nếu là Stage B, tải trọng số tốt nhất từ Stage A lên trước
    if args.stage == 'B':
        stage_a_path = "checkpoints/ecapa_stage_A_freeze/best_model.pt"
        if os.path.exists(stage_a_path):
            print(f"[INFO] Load trọng số từ Stage A: {stage_a_path}")
            model.load_state_dict(torch.load(stage_a_path))
        else:
            print("[ERROR] Bạn phải chạy Stage A trước khi chạy Stage B!")
            sys.exit(1)
            
    # 4. Cấu hình Optimizer theo chiến lược (Differential Learning Rate)
    if args.stage == 'A':
        # Stage A: Chỉ train Head
        optimizer = optim.Adam(model.head.parameters(), lr=1e-3)
    else:
        # Stage B: Train cả mạng với LR khác nhau
        optimizer = optim.Adam([
            {'params': model.ecapa.parameters(), 'lr': 1e-5}, # Backbone học rất chậm
            {'params': model.head.parameters(), 'lr': 1e-4}   # Head học bình thường
        ])

    criterion = nn.CrossEntropyLoss()
    
    # 5. Huấn luyện
    best_eer = float('inf')
    val_pairs_path = "datasets/preprocessing/val_pairs.csv"
    
    # Khởi tạo thông số Early Stopping
    patience = 5
    epochs_no_improve = 0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for waveforms, labels in pbar:
            waveforms, labels = waveforms.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(waveforms)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'Loss': loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation EER cuối mỗi epoch
        val_eer = validate_eer(model, val_pairs_path, "datasets/preprocessing", device)
        
        print(f"-> Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val EER: {val_eer*100:.4f}%")
        
        # Lưu best model & Cập nhật Early Stopping
        if val_eer < best_eer:
            best_eer = val_eer
            epochs_no_improve = 0  # Reset bộ đếm vì mô hình vừa tốt lên
            
            save_path = os.path.join(save_dir, "best_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"[*] Đã lưu mô hình tốt nhất mới (EER: {val_eer*100:.4f}%) tại {save_path}")
        else:
            epochs_no_improve += 1
            print(f"[!] Val EER không cải thiện. Patience: {epochs_no_improve}/{patience}")
            
        # Kích hoạt Early Stopping nếu hết kiên nhẫn
        if epochs_no_improve >= patience:
            print(f"\n[INFO] Kích hoạt Early Stopping tại Epoch {epoch}! Val EER đã không giảm trong {patience} epoch liên tiếp.")
            print(f"[INFO] Kết thúc huấn luyện sớm. Mô hình tốt nhất đã lưu có EER: {best_eer*100:.4f}%")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune ECAPA-TDNN")
    parser.add_argument('--stage', type=str, required=True, choices=['A', 'B'], help="A (Freeze) or B (Finetune All)")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    
    train(args)