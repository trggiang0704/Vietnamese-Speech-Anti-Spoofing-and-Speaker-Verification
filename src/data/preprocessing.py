import os
import hashlib
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

# ==========================================
# CẤU HÌNH THƯ MỤC VÀ SIÊU THAM SỐ
# ==========================================
PROJECT_ROOT = r"D:\Đồ Án\VSASV"
RAW_DIR = os.path.join(PROJECT_ROOT, "datasets", "raw", "VSASV_FULL")
METADATA_PATH = os.path.join(RAW_DIR, "metadata.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "datasets", "preprocessing")

TARGET_SR = 16000
TARGET_DURATION = 3.0  # seconds
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)

# ==========================================
# HÀM HỖ TRỢ
# ==========================================
def get_input_path(row):
    """Đường dẫn thực tế ghép thẳng RAW_DIR với cột filename"""
    rel_path = str(row['filename'])
    # Chuẩn hóa dấu gạch chéo (/ hoặc \) theo hệ điều hành đang dùng (Windows)
    rel_path = rel_path.replace('/', os.sep).replace('\\', os.sep)
    return os.path.join(RAW_DIR, rel_path)

def compute_file_hash(filepath):
    """Tính mã băm SHA-256 để phát hiện file trùng lặp byte-identical."""
    hasher = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None

def process_single_audio(task):
    """Load -> Mono -> 16kHz -> Trim/Pad 3.0s -> Peak Normalize -> Save."""
    input_path, output_path = task
    try:
        if not os.path.exists(input_path):
            return (input_path, False, "File không tồn tại")

        # 1. Load + Mono + 16kHz
        wav, _ = librosa.load(input_path, sr=TARGET_SR, mono=True)

        # 2. Trim hoặc Pad cố định 3 giây
        if len(wav) > TARGET_SAMPLES:
            wav = wav[:TARGET_SAMPLES]
        else:
            wav = np.pad(wav, (0, TARGET_SAMPLES - len(wav)), mode='constant')

        # 3. Peak Normalization
        max_val = np.max(np.abs(wav))
        if max_val > 0:
            wav = wav / max_val

        # 4. Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, wav, TARGET_SR)
        return (input_path, True, None)
    except Exception as e:
        return (input_path, False, str(e))

def greedy_speaker_split(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Chia tập độc lập theo ID Speaker (Speaker-Independent)"""
    print(f"\n[INFO] Bắt đầu chia tập Speaker-Independent (Train={train_ratio}, Val={val_ratio}, Test={test_ratio})...")
    speaker_counts = df['speaker'].value_counts().to_dict()

    sorted_speakers = sorted(speaker_counts.items(), key=lambda x: x[1], reverse=True)
    total_files = len(df)
    targets = {
        'train': train_ratio * total_files,
        'val': val_ratio * total_files,
        'test': test_ratio * total_files
    }

    current_counts = {'train': 0, 'val': 0, 'test': 0}
    speaker_to_split = {}

    for spk, count in sorted_speakers:
        best_split = min(targets.keys(), key=lambda k: current_counts[k] / targets[k] if targets[k] > 0 else float('inf'))
        speaker_to_split[spk] = best_split
        current_counts[best_split] += count

    df['split'] = df['speaker'].map(speaker_to_split)

    print("\n[RESULT] Phân phối sau khi chia tập:")
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        print(f"  - {split.upper()}: {len(split_df):,} files ({len(split_df)/total_files*100:.2f}%) - {split_df['speaker'].nunique()} speakers")

    return df

# ==========================================
# LUỒNG THỰC THI CHÍNH
# ==========================================
def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print(f"[INFO] Đang đọc metadata từ: {METADATA_PATH}")
    df = pd.read_csv(METADATA_PATH)

    # 1. Deduplication (Lọc file trùng lặp byte-identical)
    print("\n[INFO] Đang quét mã băm (SHA-256) để khử trùng lặp...")
    hashes = []
    sample_failed_path = None
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Hashing"):
        file_path = get_input_path(row)
        file_hash = compute_file_hash(file_path)
        hashes.append(file_hash)
        
        # Lưu lại mẫu đường dẫn lỗi đầu tiên để dễ debug
        if file_hash is None and sample_failed_path is None:
            sample_failed_path = file_path

    df['file_hash'] = hashes
    missing_count = df['file_hash'].isna().sum()
    
    if missing_count > 0:
        print(f"\n[ERROR] Không tìm thấy {missing_count} file trên ổ cứng.")
        print(f"[DEBUG] Đường dẫn lỗi mẫu: {sample_failed_path}")
        
    df = df.dropna(subset=['file_hash'])

    if len(df) == 0:
        print("\n[FATAL] Không có file nào hợp lệ để xử lý. Dừng chương trình.")
        sys.exit(1)

    initial_count = len(df)
    df = df.drop_duplicates(subset=['file_hash'], keep='first')
    dedup_count = len(df)
    print(f"[RESULT] Đã lọc bỏ {initial_count - dedup_count} file trùng. Giữ lại {dedup_count:,} file unique.")

    # 2. Chia tập Train/Val/Test
    df = greedy_speaker_split(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    # Lưu metadata từng split
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split].copy()
        split_csv_path = os.path.join(PROCESSED_DIR, f"{split}_metadata.csv")
        split_df.to_csv(split_csv_path, index=False)
        print(f"[INFO] Đã lưu metadata tập {split}: {split_csv_path}")

    # 3. Chuẩn bị tác vụ tiền xử lý audio
    print("\n[INFO] Chuẩn bị tác vụ chuẩn hóa Audio (16kHz, Mono, 3.0s, Normalize)...")
    tasks = []
    for _, row in df.iterrows():
        in_path = get_input_path(row)
        out_filename = os.path.basename(in_path)
        out_path = os.path.join(PROCESSED_DIR, row['split'], row['utt_type'], out_filename)
        tasks.append((in_path, out_path))

    # 4. Xử lý đa tiến trình
    print(f"[INFO] Bắt đầu xử lý {len(tasks):,} file âm thanh...")
    success_count = 0
    error_list = []

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_single_audio, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Audio"):
            in_path, success, err_msg = future.result()
            if success:
                success_count += 1
            else:
                error_list.append((in_path, err_msg))

    print(f"\n[INFO] Hoàn tất Phase 2!")
    print(f"[RESULT] Thành công: {success_count:,}/{len(tasks):,} files.")

    if error_list:
        error_log_path = os.path.join(PROCESSED_DIR, "error_log.txt")
        print(f"[WARNING] Có {len(error_list)} file bị lỗi. Đang ghi log tại: {error_log_path}")
        with open(error_log_path, "w", encoding='utf-8') as f:
            for path, err in error_list:
                f.write(f"{path}\t{err}\n")

if __name__ == "__main__":
    main()