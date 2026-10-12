import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def calculate_eer(y_true, y_scores):
    """Tính toán Equal Error Rate (EER)"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh

def compute_metrics(y_true, y_scores, threshold=0.5):
    """Tính các metrics cơ bản toàn cục (Global Metrics)"""
    y_pred = (np.array(y_scores) >= threshold).astype(int)
    y_true = np.array(y_true)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_scores)
    eer, eer_thresh = calculate_eer(y_true, y_scores)
    cm = confusion_matrix(y_true, y_pred)

    return {
        'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1,
        'ROC-AUC': auc, 'EER': eer, 'EER_Threshold': eer_thresh, 'CM': cm
    }

def per_spoof_evaluation(results_df, threshold=0.5):
    """
    Phân tích tỷ lệ nhận diện lỗi theo từng loại spoof (VC, Replay, Adversarial).
    DataFrame cần có: 'true_label' (1=Bonafide, 0=Spoof), 'score' (xác suất Bonafide), 'utt_type'
    """
    spoof_types = results_df[results_df['true_label'] == 0]['utt_type'].unique()
    bonafide_df = results_df[results_df['true_label'] == 1]
    
    per_spoof_metrics = {}
    for stype in spoof_types:
        spoof_df = results_df[results_df['utt_type'] == stype]
        
        # EER cho riêng cặp (Bonafide vs Specific Spoof)
        combined_df = pd.concat([bonafide_df, spoof_df])
        eer, _ = calculate_eer(combined_df['true_label'].values, combined_df['score'].values)
        
        # Spoof Recall: Khả năng bắt đúng Spoof (Score < Threshold)
        spoof_preds = (spoof_df['score'].values < threshold).astype(int)
        recall_spoof = np.sum(spoof_preds) / len(spoof_preds) if len(spoof_preds) > 0 else 0
        
        per_spoof_metrics[stype] = {'EER': eer, 'Spoof_Recall': recall_spoof}
        
    return per_spoof_metrics