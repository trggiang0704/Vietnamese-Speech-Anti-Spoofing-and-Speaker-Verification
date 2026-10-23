import os
import sys
import argparse
import torch
import torch.nn as nn
import soundfile as sf
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# Tải SpeechBrain
from speechbrain.inference.speaker import EncoderClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# ==========================================
# 1. ĐỊNH NGHĨA LẠI MODEL WRAPPER (Để load Weights)
# ==========================================
class ECAPAFineTuner(nn.Module):
    def __init__(self, num_classes, device='cuda:0'):
        super().__init__()
        self.ecapa = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            savedir="checkpoints/pretrained_ecapa",
            run_opts={"device": device}
        )
        # Giữ nguyên Head để map với state_dict
        self.head = nn.Linear(192, num_classes)
        
    def forward(self, x):
        feats = self.ecapa.mods.compute_features(x)
        feats = self.ecapa.mods.mean_var_norm(feats, torch.ones(x.shape[0], device=x.device))
        embeddings = self.ecapa.mods.embedding_model(feats).squeeze(2).squeeze(1)
        # Trong Inference Speaker Verification, ta chỉ cần embeddings (192-dim)
        return embeddings

# ==========================================
# 2. HÀM TÍNH METRICS
# ==========================================
def calculate_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh, fpr, tpr

def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Chạy đánh giá ECAPA-TDNN trên {device}")
    
    # Tạo thư mục báo cáo riêng cho từng Run
    report_dir = os.path.join("reports", args.run_name)
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(os.path.join(report_dir, "figures"), exist_ok=True)
    
    # 1. Load Model và Trọng số
    # Tổng số class là 917 (từ file Train mà bạn đã chạy log)
    model = ECAPAFineTuner(num_classes=917, device=device).to(device)
    
    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Không tìm thấy file trọng số: {args.checkpoint}")
        sys.exit(1)
        
    print(f"[INFO] Load trọng số từ: {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    # 2. Đọc file Test Pairs (10.000 cặp)
    pairs_file = "datasets/preprocessing/test_pairs.csv"
    data_dir = "datasets/preprocessing"
    df = pd.read_csv(pairs_file)
    
    cos_sim = nn.CosineSimilarity(dim=0)
    y_true = []
    y_scores = []
    
    print("[INFO] Bắt đầu trích xuất Embeddings và tính Cosine Similarity...")
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Test Inference ({args.run_name})"):
            f1_name = os.path.basename(str(row['audio1']).replace('\\', '/'))
            f2_name = os.path.basename(str(row['audio2']).replace('\\', '/'))
            
            f1_path = os.path.join(data_dir, 'test', 'bonafide', f1_name)
            f2_path = os.path.join(data_dir, 'test', 'bonafide', f2_name)
            
            wav1, _ = sf.read(f1_path)
            wav2, _ = sf.read(f2_path)
            
            sig1 = torch.tensor(wav1, dtype=torch.float32).unsqueeze(0).to(device)
            sig2 = torch.tensor(wav2, dtype=torch.float32).unsqueeze(0).to(device)
            
            emb1 = model(sig1).squeeze()
            emb2 = model(sig2).squeeze()
            
            score = cos_sim(emb1, emb2).item()
            y_scores.append(score)
            y_true.append(row['label'])
            
    # 3. Tính toán Metrics
    eer, thresh, fpr, tpr = calculate_eer(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    
    print("\n" + "="*55)
    print(f" KẾT QUẢ TEST SET: {args.run_name.upper()}")
    print("="*55)
    print(f"Speaker EER     : {eer * 100:.4f} %")
    print(f"ROC-AUC         : {auc:.4f}")
    print(f"Best Threshold  : {thresh:.4f}")
    
    # 4. Lưu Score và Hình ảnh
    df['score'] = y_scores
    df.to_csv(os.path.join(report_dir, "scores.csv"), index=False)
    
    # Lưu config
    import json
    with open(os.path.join(report_dir, "metrics.json"), 'w') as f:
        json.dump({"SV-EER": float(eer), "ROC-AUC": float(auc), "Threshold": float(thresh)}, f, indent=4)
        
    # --- Vẽ ROC Curve ---
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC (EER = {eer*100:.2f}%)")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC - Speaker Verification ({args.run_name})")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(report_dir, "figures", "roc.png"))
    plt.close()
    
    # --- Vẽ Score Distribution ---
    plt.figure(figsize=(10, 6))
    scores_target = df[df['label'] == 1]['score']
    scores_imposter = df[df['label'] == 0]['score']
    
    plt.hist(scores_target, bins=50, alpha=0.6, color='green', label='Same Speaker (Target)')
    plt.hist(scores_imposter, bins=50, alpha=0.6, color='red', label='Different Speaker (Imposter)')
    plt.axvline(thresh, color='black', linestyle='dashed', linewidth=2, label='EER Threshold')
    
    plt.title(f"Score Distribution ({args.run_name})")
    plt.xlabel("Cosine Similarity Score")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(os.path.join(report_dir, "figures", "score_distribution.png"))
    plt.close()
    
    print(f"\n[INFO] Đã xuất báo cáo tại: {report_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Fine-tuned ECAPA-TDNN")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to best_model.pt")
    parser.add_argument('--run_name', type=str, required=True, help="Folder name for reports (e.g., stage_A)")
    args = parser.parse_args()
    
    main(args)