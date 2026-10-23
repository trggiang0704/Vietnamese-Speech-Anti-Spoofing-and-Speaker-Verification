import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def generate_pairs(metadata_path, output_path, num_pairs=10000):
    """
    Tạo danh sách cặp đánh giá Speaker Verification từ file metadata.
    Chỉ sử dụng dữ liệu BONAFIDE.
    """
    print(f"[INFO] Đọc dữ liệu từ: {metadata_path}")
    df = pd.read_csv(metadata_path)
    
    # 1. Lọc KHÔNG đưa spoof vào speaker branch
    df = df[df['utt_type'] == 'bonafide'].copy()
    
    # Lấy danh sách speaker và gom nhóm file theo speaker
    speakers = df['speaker'].unique()
    speaker_files = df.groupby('speaker')['filename'].apply(list).to_dict()
    
    pairs = []
    
    # 2. Tạo Positive Pairs (Same Speaker) - Chiếm 50%
    target_pos = num_pairs // 2
    pos_count = 0
    pbar_pos = tqdm(total=target_pos, desc="Generating Positive Pairs")
    
    while pos_count < target_pos:
        spk = np.random.choice(speakers)
        files = speaker_files[spk]
        if len(files) >= 2:
            f1, f2 = np.random.choice(files, 2, replace=False)
            pairs.append({'audio1': f1, 'audio2': f2, 'label': 1, 'type': 'bonafide'})
            pos_count += 1
            pbar_pos.update(1)
            
    pbar_pos.close()
    
    # 3. Tạo Negative Pairs (Different Speakers) - Chiếm 50%
    target_neg = num_pairs - target_pos
    neg_count = 0
    pbar_neg = tqdm(total=target_neg, desc="Generating Negative Pairs")
    
    while neg_count < target_neg:
        spk1, spk2 = np.random.choice(speakers, 2, replace=False)
        f1 = np.random.choice(speaker_files[spk1])
        f2 = np.random.choice(speaker_files[spk2])
        pairs.append({'audio1': f1, 'audio2': f2, 'label': 0, 'type': 'bonafide'})
        neg_count += 1
        pbar_neg.update(1)
        
    pbar_neg.close()
    
    # 4. Trộn ngẫu nhiên (Shuffle) và lưu
    pairs_df = pd.DataFrame(pairs)
    pairs_df = pairs_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    pairs_df.to_csv(output_path, index=False)
    print(f"[RESULT] Đã lưu {len(pairs_df)} cặp tại: {output_path}")

if __name__ == "__main__":
    np.random.seed(42)
    
    # Tạo 5000 cặp cho tập Val (để tuning threshold)
    val_meta = "datasets/preprocessing/val_metadata.csv"
    val_out = "datasets/preprocessing/val_pairs.csv"
    if os.path.exists(val_meta):
        generate_pairs(val_meta, val_out, num_pairs=5000)
    else:
        print(f"[ERROR] Không tìm thấy file Val Metadata tại: {os.path.abspath(val_meta)}")
        
    # Tạo 10000 cặp cho tập Test (để đánh giá FAR/FRR cuối cùng)
    test_meta = "datasets/preprocessing/test_metadata.csv"
    test_out = "datasets/preprocessing/test_pairs.csv"
    if os.path.exists(test_meta):
        generate_pairs(test_meta, test_out, num_pairs=10000)
    else:
        print(f"[ERROR] Không tìm thấy file Test Metadata tại: {os.path.abspath(test_meta)}")