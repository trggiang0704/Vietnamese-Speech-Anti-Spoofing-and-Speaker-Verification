import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

# Thêm root path để import được các module từ src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.dataset import VSASVDataset
from src.features.logmel import LogMelSpectrogram
from src.models.resnet18 import BaselineResNet
from src.evaluation.antispoof_metrics import compute_metrics

def train():
    # 1. Đọc cấu hình (Đã fix lỗi encoding UTF-8 trên Windows)
    config_path = "configs/experiment/baseline_resnet18.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device(cfg['training']['device'] if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Bắt đầu huấn luyện trên thiết bị: {device}")
    
    # 2. Khởi tạo Data
    train_ds = VSASVDataset(cfg['data']['train_meta'], cfg['data']['processed_dir'])
    val_ds = VSASVDataset(cfg['data']['val_meta'], cfg['data']['processed_dir'])
    
    train_loader = DataLoader(train_ds, batch_size=cfg['dataloader']['batch_size'], 
                              shuffle=True, num_workers=cfg['dataloader']['num_workers'])
    val_loader = DataLoader(val_ds, batch_size=cfg['dataloader']['batch_size'], 
                            shuffle=False, num_workers=cfg['dataloader']['num_workers'])

    # 3. Khởi tạo Model & Feature Extractor
    feature_extractor = LogMelSpectrogram(
        sample_rate=cfg['feature_extraction']['sample_rate'],
        n_fft=cfg['feature_extraction']['n_fft'],
        win_length=cfg['feature_extraction']['win_length'],
        hop_length=cfg['feature_extraction']['hop_length'],
        n_mels=cfg['feature_extraction']['n_mels']
    ).to(device)
    
    model = BaselineResNet(pretrained=True).to(device)
    
    # 4. Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'], 
                           weight_decay=cfg['training']['weight_decay'])
    
    os.makedirs(cfg['training']['save_dir'], exist_ok=True)
    best_val_eer = float('inf')

    # 5. Training Loop
    epochs = cfg['training']['epochs']
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        # Dùng tqdm để hiển thị thanh tiến trình
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for waveforms, labels, _ in pbar:
            waveforms = waveforms.to(device)
            labels = labels.to(device).unsqueeze(1) # Định dạng lại shape thành (Batch, 1)
            
            optimizer.zero_grad()
            
            # Feature extraction (trên GPU) & Forward pass
            log_mels = feature_extractor(waveforms)
            logits = model(log_mels)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'Loss': loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # 6. Validation Loop
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_scores = []
        
        with torch.no_grad():
            for waveforms, labels, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                waveforms = waveforms.to(device)
                labels_gpu = labels.to(device).unsqueeze(1)
                
                log_mels = feature_extractor(waveforms)
                logits = model(log_mels)
                
                loss = criterion(logits, labels_gpu)
                val_loss += loss.item()
                
                # Áp dụng Sigmoid để ép raw logit về không gian xác suất [0, 1]
                probs = torch.sigmoid(logits)
                
                all_labels.extend(labels.numpy())
                all_scores.extend(probs.cpu().numpy().flatten())
                
        avg_val_loss = val_loss / len(val_loader)
        metrics = compute_metrics(all_labels, all_scores)
        val_eer = metrics['EER']
        
        print(f"-> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"-> Val EER: {val_eer*100:.2f}% | Val AUC: {metrics['ROC-AUC']:.4f} | Val F1: {metrics['F1']:.4f}")
        
        # 7. Lưu Checkpoint tốt nhất (Dựa trên EER thấp nhất)
        if val_eer < best_val_eer:
            best_val_eer = val_eer
            save_path = os.path.join(cfg['training']['save_dir'], "best_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"[*] Đã lưu mô hình tốt nhất mới tại: {save_path} (EER: {val_eer*100:.2f}%)")
            
if __name__ == "__main__":
    train()