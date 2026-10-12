import os
import sys
import yaml
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_curve, ConfusionMatrixDisplay

# Thêm root path để import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.dataset import VSASVDataset
from src.features.logmel import LogMelSpectrogram
from src.models.resnet18 import BaselineResNet
from src.evaluation.antispoof_metrics import compute_metrics, per_spoof_evaluation

def evaluate():
    # 1. Đọc cấu hình
    config_path = "configs/experiment/baseline_resnet18.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device(cfg['training']['device'] if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Bắt đầu đánh giá trên thiết bị: {device}")
    
    # 2. Khởi tạo Data (Tập Test)
    test_ds = VSASVDataset(cfg['data']['test_meta'], cfg['data']['processed_dir'])
    test_loader = DataLoader(test_ds, batch_size=cfg['dataloader']['batch_size'], 
                             shuffle=False, num_workers=cfg['dataloader']['num_workers'])

    # 3. Khởi tạo Model & Load Weights
    feature_extractor = LogMelSpectrogram(
        sample_rate=cfg['feature_extraction']['sample_rate'],
        n_fft=cfg['feature_extraction']['n_fft'],
        win_length=cfg['feature_extraction']['win_length'],
        hop_length=cfg['feature_extraction']['hop_length'],
        n_mels=cfg['feature_extraction']['n_mels']
    ).to(device)
    
    model = BaselineResNet(pretrained=False).to(device)
    model_path = os.path.join(cfg['training']['save_dir'], "best_model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 4. Chạy suy luận (Inference)
    all_labels = []
    all_scores = []
    all_utt_types = []
    all_filenames = []
    
    with torch.no_grad():
        for waveforms, labels, utt_types in tqdm(test_loader, desc="Đánh giá tập Test"):
            waveforms = waveforms.to(device)
            log_mels = feature_extractor(waveforms)
            logits = model(log_mels)
            probs = torch.sigmoid(logits)
            
            all_labels.extend(labels.numpy())
            all_scores.extend(probs.cpu().numpy().flatten())
            all_utt_types.extend(utt_types)

    # Lấy thêm tên file để lưu CSV
    all_filenames = test_ds.metadata['filename'].values.tolist()

    # 5. Phân tích kết quả
    results_df = pd.DataFrame({
        'filename': all_filenames,
        'true_label': all_labels,
        'score': all_scores,
        'utt_type': all_utt_types
    })

    global_metrics = compute_metrics(results_df['true_label'], results_df['score'])
    per_spoof = per_spoof_evaluation(results_df, threshold=global_metrics['EER_Threshold'])
    
    # 6. TẠO THƯ MỤC VÀ LƯU FILE / ẢNH
    report_dir = "reports/antispoof"
    os.makedirs(report_dir, exist_ok=True)
    
    # 6.1. Lưu báo cáo dạng Text
    txt_report_path = os.path.join(report_dir, "baseline_report.txt")
    with open(txt_report_path, "w", encoding="utf-8") as f:
        f.write("="*50 + "\n")
        f.write(" KẾT QUẢ ĐÁNH GIÁ BASELINE RESNET18 (TEST SET)\n")
        f.write("="*50 + "\n\n")
        f.write("--- 1. GLOBAL METRICS ---\n")
        f.write(f"EER             : {global_metrics['EER'] * 100:.4f} %\n")
        f.write(f"ROC-AUC         : {global_metrics['ROC-AUC']:.4f}\n")
        f.write(f"Accuracy        : {global_metrics['Accuracy'] * 100:.4f} %\n")
        f.write(f"F1-Score        : {global_metrics['F1']:.4f}\n")
        f.write(f"Precision       : {global_metrics['Precision']:.4f}\n")
        f.write(f"Recall          : {global_metrics['Recall']:.4f}\n")
        f.write(f"EER Threshold   : {global_metrics['EER_Threshold']:.4f}\n\n")
        
        f.write("--- 2. PER-SPOOF EVALUATION ---\n")
        for stype, mets in per_spoof.items():
            f.write(f"- {stype.upper()}:\n")
            f.write(f"  + EER          : {mets['EER'] * 100:.4f} %\n")
            f.write(f"  + Spoof Recall : {mets['Spoof_Recall'] * 100:.2f} %\n")
            
    # Đọc nội dung file text ra màn hình luôn
    with open(txt_report_path, "r", encoding="utf-8") as f:
        print(f.read())
        
    # 6.2. Lưu chi tiết toàn bộ Score ra CSV
    csv_path = os.path.join(report_dir, "baseline_test_scores.csv")
    results_df['decision'] = (results_df['score'] >= global_metrics['EER_Threshold']).astype(int)
    results_df.to_csv(csv_path, index=False)
    
    # 6.3. Vẽ và lưu ROC Curve
    fpr, tpr, _ = roc_curve(results_df['true_label'], results_df['score'], pos_label=1)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC curve (AUC = {global_metrics['ROC-AUC']:.4f})")
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(report_dir, "baseline_roc.png"))
    plt.close()

    # 6.4. Vẽ và lưu Confusion Matrix
    cm = global_metrics['CM']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Spoof', 'Bonafide'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix - Baseline')
    plt.savefig(os.path.join(report_dir, "baseline_cm.png"))
    plt.close()
    
    # 6.5. Vẽ và lưu Score Distribution (Đã fix lỗi mất ảnh)
    plt.figure(figsize=(10, 6))
    
    # Ép kiểu về numpy array để matplotlib xử lý an toàn nhất
    bonafide_scores = results_df[results_df['true_label'] == 1]['score'].values
    spoof_scores = results_df[results_df['true_label'] == 0]['score'].values
    
    # Ép buộc trục hoành (x-axis) luôn chia từ 0.0 đến 1.0 thành 50 cột
    bins_range = np.linspace(0, 1, 50)
    
    plt.hist(bonafide_scores, bins=bins_range, alpha=0.6, color='green', label='Bonafide')
    plt.hist(spoof_scores, bins=bins_range, alpha=0.6, color='red', label='Spoof')
    plt.axvline(global_metrics['EER_Threshold'], color='black', linestyle='dashed', linewidth=2, label='EER Threshold')
    
    plt.title('Score Distribution (Baseline ResNet18)')
    plt.xlabel('Probability Score')
    plt.ylabel('Count')
    plt.legend()
    
    # Thêm bbox_inches='tight' để đảm bảo ảnh không bị cắt viền
    plt.savefig(os.path.join(report_dir, "baseline_score_dist.png"), bbox_inches='tight')
    plt.close()

    print(f"\n[INFO] Đã xuất toàn bộ file đánh giá (Ảnh, CSV, Txt) vào thư mục: {report_dir}")

if __name__ == "__main__":
    evaluate()