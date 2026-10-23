import os
import sys
import torch
import soundfile as sf
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# Tải thư viện ECAPA-TDNN từ SpeechBrain
from speechbrain.inference.speaker import EncoderClassifier

# Thêm root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# ==========================================
# CẤU HÌNH
# ==========================================
DATA_DIR = "datasets/preprocessing"
PAIRS_FILE = "datasets/preprocessing/test_pairs.csv"
SPLIT = "test"  # Vì test_pairs.csv lấy từ test_metadata
REPORT_DIR = "reports/speaker_verification"

def calculate_eer(y_true, y_scores):
    """Tính toán Equal Error Rate (EER)"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh, fpr, tpr

def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(os.path.join(REPORT_DIR, "figures"), exist_ok=True)

    # 1. Khởi tạo mô hình
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Tải mô hình ECAPA-TDNN pretrained (SpeechBrain) lên {device}...")
    
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        savedir="checkpoints/pretrained_ecapa",
        run_opts={"device": device}
    )

    # 2. Đọc danh sách cặp đánh giá
    print(f"[INFO] Đọc file cặp đánh giá Test: {PAIRS_FILE}")
    df = pd.read_csv(PAIRS_FILE)

    cos_sim = torch.nn.CosineSimilarity(dim=0)
    y_true = []
    y_scores = []

    # 3. Trích xuất đặc trưng (Embeddings) và tính điểm
    print("[INFO] Bắt đầu trích xuất Embedding và tính Cosine Similarity...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Inference ECAPA"):
        # SỬA LỖI ĐƯỜNG DẪN Ở ĐÂY: Chỉ lấy basename của audio1 và audio2
        f1_name = os.path.basename(str(row['audio1']).replace('\\', '/'))
        f2_name = os.path.basename(str(row['audio2']).replace('\\', '/'))
        
        f1_path = os.path.join(DATA_DIR, SPLIT, row['type'], f1_name)
        f2_path = os.path.join(DATA_DIR, SPLIT, row['type'], f2_name)

        if not os.path.exists(f1_path):
            print(f"\n[ERROR] Không tìm thấy file: {f1_path}")
            sys.exit(1)
        if not os.path.exists(f2_path):
            print(f"\n[ERROR] Không tìm thấy file: {f2_path}")
            sys.exit(1)

        # Sử dụng soundfile để đọc an toàn thay vì torchaudio.load
        wav1, _ = sf.read(f1_path)
        wav2, _ = sf.read(f2_path)

        # Chuyển đổi Numpy Array thành Torch Tensor, thêm chiều Channel (1, Time) và đưa lên GPU
        sig1 = torch.tensor(wav1, dtype=torch.float32).unsqueeze(0).to(device)
        sig2 = torch.tensor(wav2, dtype=torch.float32).unsqueeze(0).to(device)

        # Trích xuất 192-dimensional Speaker Embeddings
        with torch.no_grad():
            emb1 = classifier.encode_batch(sig1).squeeze()
            emb2 = classifier.encode_batch(sig2).squeeze()

        # Tính độ tương đồng Cosine [-1, 1]
        score = cos_sim(emb1, emb2).item()
        
        y_scores.append(score)
        y_true.append(row['label'])

    # 4. Tính toán Metrics
    eer, thresh, fpr, tpr = calculate_eer(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    
    print("\n" + "="*55)
    print(" KẾT QUẢ ĐÁNH GIÁ ECAPA-TDNN PRETRAINED (SPEAKER BRANCH)")
    print("="*55)
    print(f"Speaker EER     : {eer * 100:.4f} %")
    print(f"ROC-AUC         : {auc:.4f}")
    print(f"Best Threshold  : {thresh:.4f}")
    
    # 5. Lưu báo cáo và hình ảnh
    df['score'] = y_scores
    df.to_csv(os.path.join(REPORT_DIR, "scores.csv"), index=False)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='green', lw=2, label=f"ROC (EER = {eer*100:.2f}%)")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC - Speaker Verification (ECAPA-TDNN)")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(REPORT_DIR, "figures", "roc.png"))
    plt.close()
    
    import json
    with open(os.path.join(REPORT_DIR, "metrics.json"), 'w') as f:
        json.dump({"SV-EER": float(eer), "ROC-AUC": float(auc), "Threshold": float(thresh)}, f, indent=4)
        
    print(f"\n[INFO] Đã lưu báo cáo (scores.csv, roc.png, metrics.json) tại {REPORT_DIR}")

if __name__ == "__main__":
    main()