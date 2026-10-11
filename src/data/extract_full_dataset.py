
import re
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm


# =========================================================
# CONFIG
# =========================================================

SEED = 42  # Lưu để đảm bảo tính nhất quán cho các bước sau

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HF_CACHE_DIR = Path(r"F:\hf_cache")

DATASET_NAME = "datasets--hustep-lab--VSASV-Dataset"

SNAPSHOTS_DIR = (
    HF_CACHE_DIR / "hub" / DATASET_NAME / "snapshots"
)

OUTPUT_DIR = PROJECT_ROOT / "datasets" / "raw" / "VSASV_FULL"
AUDIO_DIR = OUTPUT_DIR / "audio"

METADATA_PATH = OUTPUT_DIR / "metadata.csv"
ERRORS_PATH = OUTPUT_DIR / "extraction_errors.csv"

# True: bỏ qua file đã tồn tại khi chạy lại.
# Chỉ dùng khi thư mục output là kết quả của lần chạy trước
# và metadata.csv vẫn còn nguyên vẹn.
SKIP_EXISTING = False


# =========================================================
# HELPERS
# =========================================================

def safe_folder_name(value):
    """Chuyển utt_type thành tên thư mục an toàn."""
    value = str(value).strip()

    if not value or value.lower() in {"nan", "none"}:
        value = "unknown"

    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value)
    return value[:100]


def value_to_string(value):
    """Chuẩn hóa giá trị metadata để ghi CSV."""
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value)


def get_audio_info(audio_obj):
    """
    Đọc audio theo cấu trúc mà code cũ đang sử dụng:
    {"array": ..., "sampling_rate": ...}
    """
    if not isinstance(audio_obj, dict):
        raise ValueError("Audio field không phải dictionary")

    if audio_obj.get("array") is None:
        raise ValueError("Audio array bị thiếu")

    array = np.asarray(audio_obj["array"])

    if array.size == 0:
        raise ValueError("Audio array rỗng")

    if array.ndim not in (1, 2):
        raise ValueError(f"Audio có số chiều không hợp lệ: {array.ndim}")

    sr = int(audio_obj["sampling_rate"])

    if sr <= 0:
        raise ValueError(f"Sample rate không hợp lệ: {sr}")

    if not np.isfinite(array).all():
        raise ValueError("Audio chứa NaN hoặc Inf")

    # soundfile cần dữ liệu âm thanh hợp lệ.
    # Không tự ý resample hoặc normalize trong bước trích xuất.
    if np.issubdtype(array.dtype, np.integer):
        array = array.astype(np.float32)
    else:
        array = array.astype(np.float32, copy=False)

    return array, sr


# =========================================================
# FIND SNAPSHOT
# =========================================================

if not SNAPSHOTS_DIR.exists():
    raise FileNotFoundError(
        f"Không tìm thấy thư mục snapshots: {SNAPSHOTS_DIR}"
    )

snapshots = sorted(
    p for p in SNAPSHOTS_DIR.iterdir() if p.is_dir()
)

if not snapshots:
    raise FileNotFoundError(
        f"Không có snapshot nào trong: {SNAPSHOTS_DIR}"
    )

# Chọn snapshot mới nhất theo thời gian sửa đổi.
# In ra đường dẫn để có thể kiểm tra snapshot thực sự được dùng.
snapshot_dir = max(snapshots, key=lambda p: p.stat().st_mtime)
parquet_dir = snapshot_dir / "data"

if not parquet_dir.exists():
    raise FileNotFoundError(
        f"Không tìm thấy thư mục data: {parquet_dir}"
    )

parquet_files = sorted(parquet_dir.glob("*.parquet"))

if not parquet_files:
    raise FileNotFoundError(
        f"Không tìm thấy file Parquet trong: {parquet_dir}"
    )

print("=" * 72)
print("VSASV FULL DATASET EXTRACTION")
print("=" * 72)
print(f"Snapshot: {snapshot_dir}")
print(f"Parquet files: {len(parquet_files)}")
print(f"Output: {OUTPUT_DIR}")
print("Mode: extract all rows; no class filtering; no sample quota")
print("=" * 72)

AUDIO_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# EXTRACT ALL ROWS
# =========================================================

metadata_rows = []
error_rows = []
global_index = 0
success_count = 0
failed_count = 0

for parquet_path in tqdm(parquet_files, desc="Parquet files"):

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as exc:
        error_rows.append({
            "parquet_file": parquet_path.name,
            "row_index": "",
            "reason": f"Cannot read parquet: {exc}"
        })
        failed_count += 1
        continue

    # Không lọc utt_type: giữ lại tất cả các loại có trong nguồn.
    for row_index, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=parquet_path.name,
        leave=False
    ):
        global_index += 1

        utt_type = value_to_string(row.get("utt_type", ""))
        speaker = value_to_string(row.get("label", ""))

        audio_obj = row.get("audio", None)

        try:
            array, sr = get_audio_info(audio_obj)

            # Mỗi bản ghi có ID riêng để tránh ghi đè filename.
            audio_id = f"utt_{global_index:09d}"
            class_folder = safe_folder_name(utt_type)

            relative_path = (
                Path("audio") / class_folder / f"{audio_id}.wav"
            )
            save_path = OUTPUT_DIR / relative_path
            save_path.parent.mkdir(parents=True, exist_ok=True)

            if SKIP_EXISTING and save_path.exists():
                # Chỉ an toàn khi metadata của lần chạy trước
                # vẫn đầy đủ; mặc định SKIP_EXISTING=False.
                pass
            else:
                sf.write(save_path, array, sr, format="WAV")

            duration = len(array) / sr

            metadata_rows.append({
                "audio_id": audio_id,
                "filename": relative_path.as_posix(),
                "speaker": speaker,
                "label": value_to_string(row.get("label", "")),
                "utt_type": utt_type,
                "sample_rate": sr,
                "num_samples": len(array),
                "duration_seconds": duration,
                "source_parquet": parquet_path.name,
                "source_row_index": row_index,
            })

            success_count += 1

        except Exception as exc:
            error_rows.append({
                "parquet_file": parquet_path.name,
                "row_index": row_index,
                "utt_type": utt_type,
                "speaker": speaker,
                "reason": str(exc)
            })
            failed_count += 1


# =========================================================
# SAVE MANIFESTS
# =========================================================

metadata_columns = [
    "audio_id",
    "filename",
    "speaker",
    "label",
    "utt_type",
    "sample_rate",
    "num_samples",
    "duration_seconds",
    "source_parquet",
    "source_row_index",
]

metadata_df = pd.DataFrame(
    metadata_rows,
    columns=metadata_columns
)

metadata_df.to_csv(
    METADATA_PATH,
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame(error_rows).to_csv(
    ERRORS_PATH,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# SUMMARY
# =========================================================

print("\n" + "=" * 72)
print("EXTRACTION SUMMARY")
print("=" * 72)
print(f"Snapshot used : {snapshot_dir}")
print(f"Parquet files : {len(parquet_files)}")
print(f"Rows processed: {global_index}")
print(f"Audio saved   : {success_count}")
print(f"Failed rows   : {failed_count}")
print(f"Metadata      : {METADATA_PATH}")
print(f"Errors        : {ERRORS_PATH}")

if not metadata_df.empty:
    print("\nDistribution by utt_type:")
    print(
        metadata_df["utt_type"]
        .replace("", "unknown")
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nDistribution by sample rate:")
    print(metadata_df["sample_rate"].value_counts().to_string())

print("=" * 72)

if failed_count:
    print(
        "WARNING: Có bản ghi chưa trích xuất thành công. "
        "Hãy kiểm tra extraction_errors.csv trước khi audit."
    )
else:
    print("All processed rows were extracted successfully.")