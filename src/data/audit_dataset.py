from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "datasets" / "raw" / "VSASV_FULL"
METADATA_PATH = DATASET_DIR / "metadata.csv"
AUDIO_DIR = DATASET_DIR / "audio"

REPORT_DIR = PROJECT_ROOT / "reports" / "dataset"
FIGURE_DIR = REPORT_DIR / "figures"

# Content hash uses SHA-256 over file bytes. Exact duplicate files
# with different WAV headers/encoding may not be detected as duplicates.
HASH_ALGORITHM = "sha256"

# Audio files to scan. SoundFile can inspect WAV/FLAC/OGG and other
# formats supported by the installed libsndfile.
AUDIO_EXTENSIONS = {
    ".wav", ".flac", ".ogg", ".aiff", ".aif", ".au"
}

# Canonical class-name rules. Unmatched types are retained and reported
# as "other", rather than silently discarded.
BONAFIDE_TERMS = {
    "bonafide", "bona_fide", "bona-fide", "real", "genuine", "human"
}
VC_TERMS = {
    "vc", "voice_conversion", "voice-conversion", "voice conversion"
}
REPLAY_TERMS = {"replay", "replay_attack", "replay-attack"}
TTS_TERMS = {
    "tts", "text_to_speech", "text-to-speech", "text to speech",
    "speech_synthesis", "speech-synthesis", "synthesis"
}
ADVERSARIAL_TERMS = {
    "adversarial", "adversarial_attack", "adversarial-attack",
    "adversarial attack"
}

SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "dev": "validation",
    "development": "validation",
    "test": "test",
    "testing": "test",
}


# ============================================================
# HELPERS
# ============================================================

def norm(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def normalized_key(value) -> str:
    return re.sub(r"[\s\-]+", "_", norm(value).lower())


def canonical_spoof_group(utt_type: str) -> str:
    """Map known utt_type spellings to broad classes; preserve unknowns."""
    key = normalized_key(utt_type)
    compact = key.replace("_", "")

    if key in BONAFIDE_TERMS or compact in {"bonafide", "genuine"}:
        return "bonafide"
    if key in VC_TERMS or compact in {"voiceconversion", "vc"}:
        return "voice_conversion"
    if key in REPLAY_TERMS or compact.startswith("replay"):
        return "replay"
    if key in TTS_TERMS or compact.startswith("tts"):
        return "tts"
    if key in ADVERSARIAL_TERMS or compact.startswith("adversarial"):
        return "adversarial_attack"
    return "other"


def infer_split(row: pd.Series) -> str:
    """
    Infer split only when metadata contains a split column or path
    clearly includes train/val/dev/test. Do not invent a split.
    """
    for column in ("split", "subset", "partition", "set"):
        if column in row.index:
            candidate = normalized_key(row[column])
            if candidate in SPLIT_ALIASES:
                return SPLIT_ALIASES[candidate]

    path_value = norm(row.get("filename", ""))
    parts = [normalized_key(p) for p in Path(path_value).parts]
    for part in parts:
        if part in SPLIT_ALIASES:
            return SPLIT_ALIASES[part]
    return "unspecified"


def resolve_audio_path(filename: str) -> Path | None:
    if not filename:
        return None
    path = Path(filename)
    if path.is_absolute():
        return path
    # Metadata produced by the extractor uses paths relative to DATASET_DIR.
    candidate = DATASET_DIR / path
    if candidate.exists():
        return candidate
    # Also support a filename/path relative to the audio folder.
    candidate2 = AUDIO_DIR / path
    if candidate2.exists():
        return candidate2
    return candidate


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while True:
            chunk = file_handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def safe_float(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def save_bar_chart(series: pd.Series, title: str, xlabel: str,
                   ylabel: str, output_path: Path, rotate=35):
    fig, ax = plt.subplots(figsize=(11, 6))
    if len(series) == 0:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.set_axis_off()
    else:
        series.plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=rotate)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


# ============================================================
# MAIN AUDIT
# ============================================================

def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy metadata: {METADATA_PATH}\n"
            "Hãy kiểm tra DATASET_DIR/METADATA_PATH trong phần CONFIG."
        )
    if not AUDIO_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục audio: {AUDIO_DIR}\n"
            "Hãy kiểm tra AUDIO_DIR trong phần CONFIG."
        )

    print("=" * 72)
    print("VSASV DATASET AUDIT")
    print(f"Dataset : {DATASET_DIR}")
    print(f"Metadata: {METADATA_PATH}")
    print(f"Audio   : {AUDIO_DIR}")
    print(f"Reports : {REPORT_DIR}")
    print("=" * 72)

    metadata = pd.read_csv(METADATA_PATH, keep_default_na=False)
    metadata.columns = [str(c).strip() for c in metadata.columns]

    required_cols = {"filename"}
    missing_cols = required_cols - set(metadata.columns)
    if missing_cols:
        raise ValueError(
            f"metadata.csv thiếu cột bắt buộc: {sorted(missing_cols)}. "
            f"Các cột hiện có: {list(metadata.columns)}"
        )

    # Support both the current extractor's schema and modest variations.
    if "utt_type" not in metadata.columns:
        if "label" in metadata.columns:
            metadata["utt_type"] = metadata["label"]
        else:
            metadata["utt_type"] = ""

    if "speaker" not in metadata.columns:
        if "speaker_id" in metadata.columns:
            metadata["speaker"] = metadata["speaker_id"]
        elif "label" in metadata.columns:
            metadata["speaker"] = metadata["label"]
        else:
            metadata["speaker"] = ""

    if "label" not in metadata.columns:
        metadata["label"] = metadata["utt_type"]

    metadata["utt_type"] = metadata["utt_type"].map(norm)
    metadata["label"] = metadata["label"].map(norm)
    metadata["speaker"] = metadata["speaker"].map(norm)
    metadata["split_inferred"] = metadata.apply(infer_split, axis=1)
    metadata["spoof_group"] = metadata["utt_type"].map(canonical_spoof_group)

    # Discover actual audio files on disk.
    audio_paths = sorted(
        p for p in AUDIO_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )
    disk_relative = {
        p.resolve(): p.relative_to(DATASET_DIR).as_posix()
        for p in audio_paths
    }

    # Audit every metadata row's path, audio properties and byte hash.
    audit_rows = []
    hash_to_paths = defaultdict(list)
    seen_paths = Counter()
    valid_count = 0
    error_count = 0
    missing_count = 0

    for row_idx, row in tqdm(
        metadata.iterrows(), total=len(metadata), desc="Auditing metadata audio"
    ):
        filename = norm(row.get("filename", ""))
        path = resolve_audio_path(filename)
        speaker = norm(row.get("speaker", ""))
        utt_type = norm(row.get("utt_type", ""))
        split = norm(row.get("split_inferred", "unspecified"))

        status = "ok"
        error_message = ""
        sample_rate = None
        channels = None
        frames = None
        duration = None
        file_size = None
        digest = ""

        if not filename or path is None or not path.exists():
            status = "missing_file"
            error_message = "Metadata path does not resolve to an existing file"
            missing_count += 1
            error_count += 1
        else:
            resolved = path.resolve()
            seen_paths[str(resolved)] += 1
            try:
                info = sf.info(str(path))
                sample_rate = int(info.samplerate)
                channels = int(info.channels)
                frames = int(info.frames)
                duration = float(info.duration)
                file_size = int(path.stat().st_size)

                if sample_rate <= 0 or frames <= 0 or duration <= 0:
                    status = "invalid_audio_info"
                    error_message = (
                        f"Invalid properties: sr={sample_rate}, "
                        f"frames={frames}, duration={duration}"
                    )
                    error_count += 1
                else:
                    digest = sha256_file(path)
                    hash_to_paths[digest].append(str(resolved))
                    valid_count += 1
            except Exception as exc:
                status = "read_error"
                error_message = str(exc)
                error_count += 1

        audit_rows.append({
            "metadata_row": int(row_idx),
            "filename": filename,
            "resolved_path": str(path) if path else "",
            "file_status": status,
            "error_message": error_message,
            "utt_type": utt_type,
            "spoof_group": norm(row.get("spoof_group", "")),
            "label": norm(row.get("label", "")),
            "speaker": speaker,
            "split": split,
            "sample_rate": sample_rate,
            "channels": channels,
            "frames": frames,
            "duration_seconds": duration,
            "file_size_bytes": file_size,
            "sha256": digest,
        })

    audit_df = pd.DataFrame(audit_rows)
    audit_csv = REPORT_DIR / "audio_file_audit.csv"
    audit_df.to_csv(audit_csv, index=False, encoding="utf-8-sig")

    # Exact duplicate file detection (identical file bytes).
    duplicate_rows = []
    for digest, paths in hash_to_paths.items():
        if len(paths) > 1:
            for path in paths:
                duplicate_rows.append({
                    "sha256": digest,
                    "duplicate_count": len(paths),
                    "path": path,
                })
    duplicate_df = pd.DataFrame(
        duplicate_rows, columns=["sha256", "duplicate_count", "path"]
    )
    duplicate_df.to_csv(
        REPORT_DIR / "duplicate_audio.csv", index=False, encoding="utf-8-sig"
    )

    # Duplicate references in metadata (same resolved path repeated).
    repeated_path_rows = []
    for path, count in seen_paths.items():
        if count > 1:
            repeated_path_rows.append({
                "resolved_path": path,
                "metadata_reference_count": count,
            })

    # Compare metadata-listed files with files physically found under audio/.
    metadata_existing_paths = {
        str(p.resolve())
        for p in (resolve_audio_path(norm(x)) for x in metadata["filename"])
        if p is not None and p.exists()
    }
    disk_resolved_paths = {str(p.resolve()) for p in audio_paths}
    unreferenced_disk_files = sorted(disk_resolved_paths - metadata_existing_paths)
    missing_from_disk_count = len(unreferenced_disk_files)

    # Label distribution: exact label and utt_type values.
    label_dist = (
        metadata.assign(label=metadata["label"].replace("", "(missing)"))
        .groupby("label", dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
    )
    label_dist["percentage"] = (
        label_dist["count"] / max(len(metadata), 1) * 100
    )
    label_dist.to_csv(
        REPORT_DIR / "label_distribution.csv", index=False, encoding="utf-8-sig"
    )

    utt_type_dist = (
        metadata.assign(utt_type=metadata["utt_type"].replace("", "(missing)"))
        .groupby(["utt_type", "spoof_group"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
    )
    utt_type_dist["percentage"] = (
        utt_type_dist["count"] / max(len(metadata), 1) * 100
    )

    # Broad spoof distribution. Bonafide and each spoof subclass are reported.
    spoof_dist = (
        metadata.groupby("spoof_group", dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
    )
    spoof_dist["percentage"] = (
        spoof_dist["count"] / max(len(metadata), 1) * 100
    )
    spoof_dist.to_csv(
        REPORT_DIR / "spoof_type_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Speaker counts and number of utterances per speaker.
    known_speakers = metadata[metadata["speaker"].str.strip() != ""].copy()
    speaker_dist = (
        known_speakers.groupby("speaker")
        .size()
        .rename("utterance_count")
        .reset_index()
        .sort_values(["utterance_count", "speaker"], ascending=[False, True])
    )
    speaker_dist.to_csv(
        REPORT_DIR / "speaker_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    speaker_counts = speaker_dist["utterance_count"]
    speaker_stats = {
        "speaker_count_nonempty": int(known_speakers["speaker"].nunique()),
        "records_with_missing_speaker": int(
            (metadata["speaker"].str.strip() == "").sum()
        ),
        "utterances_per_speaker_min": (
            int(speaker_counts.min()) if len(speaker_counts) else None
        ),
        "utterances_per_speaker_median": (
            float(speaker_counts.median()) if len(speaker_counts) else None
        ),
        "utterances_per_speaker_mean": (
            float(speaker_counts.mean()) if len(speaker_counts) else None
        ),
        "utterances_per_speaker_max": (
            int(speaker_counts.max()) if len(speaker_counts) else None
        ),
    }

    # Duration statistics from readable files.
    good_audio = audit_df[
        (audit_df["file_status"] == "ok")
        & audit_df["duration_seconds"].notna()
    ].copy()
    duration_values = good_audio["duration_seconds"].astype(float)

    duration_stats_rows = []
    if len(duration_values):
        overall_stats = duration_values.describe(
            percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
        )
        for stat_name, stat_value in overall_stats.items():
            duration_stats_rows.append({
                "group_by": "overall",
                "group": "all",
                "statistic": stat_name,
                "value": float(stat_value),
            })

        for utt_type, group in good_audio.groupby("utt_type", dropna=False):
            values = group["duration_seconds"].astype(float)
            for stat_name, stat_value in values.describe(
                percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]
            ).items():
                duration_stats_rows.append({
                    "group_by": "utt_type",
                    "group": utt_type if utt_type else "(missing)",
                    "statistic": stat_name,
                    "value": float(stat_value),
                })

    duration_stats_df = pd.DataFrame(
        duration_stats_rows,
        columns=["group_by", "group", "statistic", "value"],
    )
    duration_stats_df.to_csv(
        REPORT_DIR / "duration_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Sample-rate and channel distributions are useful for preprocessing design.
    sample_rate_counts = (
        good_audio["sample_rate"].dropna().astype(int).value_counts().sort_index()
    )
    channel_counts = (
        good_audio["channels"].dropna().astype(int).value_counts().sort_index()
    )

    # Split distribution. "unspecified" means the metadata/path has no split;
    # the audit script deliberately does not create one.
    split_dist = (
        metadata["split_inferred"].value_counts(dropna=False).to_dict()
    )

    # Speaker overlap by split, only meaningful for explicit/inferred split names.
    split_names = {"train", "validation", "test"}
    split_speaker_sets = {}
    for split_name in split_names:
        split_speaker_sets[split_name] = set(
            metadata.loc[
                (metadata["split_inferred"] == split_name)
                & (metadata["speaker"].str.strip() != ""),
                "speaker",
            ].tolist()
        )

    overlap_rows = []
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        overlap = sorted(split_speaker_sets[left] & split_speaker_sets[right])
        overlap_rows.append({
            "split_a": left,
            "split_b": right,
            "overlap_speaker_count": len(overlap),
            "overlap_speakers": ";".join(overlap),
            "status": (
                "checked"
                if split_dist.get(left, 0) and split_dist.get(right, 0)
                else "not_checkable_no_explicit_split"
            ),
        })
    overlap_df = pd.DataFrame(overlap_rows)
    overlap_df.to_csv(
        REPORT_DIR / "speaker_overlap.csv", index=False, encoding="utf-8-sig"
    )

    # Check metadata's stored sr/duration against file inspection, if columns exist.
    consistency_rows = []
    if "sample_rate" in metadata.columns:
        for idx, audit_row in audit_df.iterrows():
            stored = safe_float(metadata.iloc[idx].get("sample_rate"))
            actual = safe_float(audit_row.get("sample_rate"))
            if stored is not None and actual is not None and int(stored) != int(actual):
                consistency_rows.append({
                    "metadata_row": idx,
                    "filename": audit_row["filename"],
                    "field": "sample_rate",
                    "metadata_value": stored,
                    "actual_file_value": actual,
                })
    if "duration_seconds" in metadata.columns:
        for idx, audit_row in audit_df.iterrows():
            stored = safe_float(metadata.iloc[idx].get("duration_seconds"))
            actual = safe_float(audit_row.get("duration_seconds"))
            if stored is not None and actual is not None and abs(stored - actual) > 0.02:
                consistency_rows.append({
                    "metadata_row": idx,
                    "filename": audit_row["filename"],
                    "field": "duration_seconds",
                    "metadata_value": stored,
                    "actual_file_value": actual,
                })
    pd.DataFrame(
        consistency_rows,
        columns=[
            "metadata_row", "filename", "field",
            "metadata_value", "actual_file_value",
        ],
    ).to_csv(
        REPORT_DIR / "extraction_metadata_checks.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Figures.
    save_bar_chart(
        label_dist.set_index("label")["count"],
        "VSASV label distribution",
        "Label",
        "Audio count",
        FIGURE_DIR / "label_distribution.png",
    )
    save_bar_chart(
        spoof_dist.set_index("spoof_group")["count"],
        "VSASV broad spoof-type distribution",
        "Spoof type / class",
        "Audio count",
        FIGURE_DIR / "spoof_distribution.png",
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    if len(duration_values):
        ax.hist(duration_values, bins=50)
        ax.set_title("VSASV audio duration distribution")
        ax.set_xlabel("Duration (seconds)")
        ax.set_ylabel("Audio count")
        ax.grid(axis="y", alpha=0.25)
    else:
        ax.text(0.5, 0.5, "No readable audio durations", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "duration_distribution.png", dpi=160)
    plt.close(fig)

    # Class balance.
    count_by_group = {
        str(row["spoof_group"]): int(row["count"])
        for _, row in spoof_dist.iterrows()
    }
    bonafide_count = count_by_group.get("bonafide", 0)
    spoof_count = sum(
        count for group, count in count_by_group.items()
        if group != "bonafide"
    )
    spoof_to_bonafide_ratio = (
        float(spoof_count / bonafide_count) if bonafide_count else None
    )

    # Audio errors and duplicate statistics.
    duplicate_group_count = int(
        (duplicate_df["duplicate_count"].drop_duplicates() > 1).sum()
    ) if not duplicate_df.empty else 0

    # Full summary JSON.
    summary = {
        "dataset_name": "VSASV_FULL",
        "dataset_dir": str(DATASET_DIR.resolve()),
        "metadata_path": str(METADATA_PATH.resolve()),
        "audio_dir": str(AUDIO_DIR.resolve()),
        "metadata_row_count": int(len(metadata)),
        "audio_files_found_on_disk": int(len(audio_paths)),
        "audio_rows_readable": int(valid_count),
        "audio_rows_with_errors": int(error_count),
        "missing_audio_files_referenced_by_metadata": int(missing_count),
        "disk_audio_files_not_referenced_by_metadata": int(missing_from_disk_count),
        "metadata_paths_referenced_multiple_times": int(len(repeated_path_rows)),
        "exact_duplicate_file_paths_in_duplicate_groups": int(len(duplicate_df)),
        "exact_duplicate_hash_groups": int(
            duplicate_df["sha256"].nunique() if not duplicate_df.empty else 0
        ),
        "rows_by_split": {str(k): int(v) for k, v in split_dist.items()},
        "split_note": (
            "No split was created by this audit. If all rows are 'unspecified', "
            "train/validation/test counts and speaker-overlap checks are not "
            "available until a split is defined."
        ),
        "label_distribution": {
            str(row["label"]): int(row["count"])
            for _, row in label_dist.iterrows()
        },
        "utt_type_distribution": {
            str(row["utt_type"]): int(row["count"])
            for _, row in utt_type_dist.iterrows()
        },
        "spoof_group_distribution": count_by_group,
        "bonafide_count": int(bonafide_count),
        "spoof_count_including_unrecognized_types": int(spoof_count),
        "spoof_to_bonafide_ratio": spoof_to_bonafide_ratio,
        "spoof_to_bonafide_ratio_note": (
            "All non-bonafide groups, including 'other', are counted as spoof "
            "for this preliminary ratio. Verify the meaning of unknown utt_type "
            "values before using this number in a paper."
        ),
        "speaker_statistics": speaker_stats,
        "sample_rate_distribution": {
            str(int(k)): int(v) for k, v in sample_rate_counts.items()
        },
        "channel_distribution": {
            str(int(k)): int(v) for k, v in channel_counts.items()
        },
        "duration_seconds": {
            "count": int(len(duration_values)),
            "total_hours": (
                float(duration_values.sum() / 3600) if len(duration_values) else 0.0
            ),
            "min": float(duration_values.min()) if len(duration_values) else None,
            "max": float(duration_values.max()) if len(duration_values) else None,
            "mean": float(duration_values.mean()) if len(duration_values) else None,
            "median": float(duration_values.median()) if len(duration_values) else None,
            "std": float(duration_values.std()) if len(duration_values) > 1 else None,
        },
        "speaker_overlap": [
            {
                "split_a": r["split_a"],
                "split_b": r["split_b"],
                "overlap_speaker_count": int(r["overlap_speaker_count"]),
                "status": r["status"],
            }
            for _, r in overlap_df.iterrows()
        ],
        "configuration": {
            "hash_algorithm": HASH_ALGORITHM,
            "hash_definition": (
                "SHA-256 of file bytes; exact byte-identical copies only"
            ),
            "audio_extensions_scanned": sorted(AUDIO_EXTENSIONS),
        },
        "generated_outputs": [
            str((REPORT_DIR / "dataset_summary.json").resolve()),
            str((REPORT_DIR / "label_distribution.csv").resolve()),
            str((REPORT_DIR / "spoof_type_distribution.csv").resolve()),
            str((REPORT_DIR / "speaker_distribution.csv").resolve()),
            str((REPORT_DIR / "duration_statistics.csv").resolve()),
            str(audit_csv.resolve()),
            str((REPORT_DIR / "duplicate_audio.csv").resolve()),
            str((REPORT_DIR / "speaker_overlap.csv").resolve()),
            str((REPORT_DIR / "extraction_metadata_checks.csv").resolve()),
            str((FIGURE_DIR / "label_distribution.png").resolve()),
            str((FIGURE_DIR / "spoof_distribution.png").resolve()),
            str((FIGURE_DIR / "duration_distribution.png").resolve()),
        ],
    }

    summary_path = REPORT_DIR / "dataset_summary.json"
    with summary_path.open("w", encoding="utf-8") as file_handle:
        json.dump(summary, file_handle, ensure_ascii=False, indent=2)

    # Console summary.
    print("\n" + "=" * 72)
    print("AUDIT SUMMARY")
    print("=" * 72)
    print(f"Metadata rows                  : {len(metadata):,}")
    print(f"Audio files found on disk      : {len(audio_paths):,}")
    print(f"Readable audio rows            : {valid_count:,}")
    print(f"Rows with errors               : {error_count:,}")
    print(f"Missing audio files            : {missing_count:,}")
    print(f"Unreferenced audio files       : {missing_from_disk_count:,}")
    print(f"Bonafide                       : {bonafide_count:,}")
    print(f"Spoof (incl. unknown types)    : {spoof_count:,}")
    print(f"Spoof / Bonafide ratio         : {spoof_to_bonafide_ratio}")
    print(f"Unique speakers                : {speaker_stats['speaker_count_nonempty']:,}")
    print(f"Readable audio duration (hours): {summary['duration_seconds']['total_hours']:.2f}")
    print("\nSplit distribution:")
    for split, count in sorted(split_dist.items()):
        print(f"  {split}: {count:,}")
    print("\nSpoof group distribution:")
    print(spoof_dist.to_string(index=False))
    print(f"\nSummary JSON: {summary_path}")
    print(f"All reports : {REPORT_DIR}")
    print("=" * 72)

    if all(v == 0 for k, v in split_dist.items() if k != "unspecified"):
        print(
            "\nNOTE: Metadata does not appear to contain train/validation/test "
            "splits. This is expected for the full extracted source. "
            "Speaker overlap cannot be meaningfully evaluated until a split "
            "is created."
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nAUDIT FAILED: {exc}", file=sys.stderr)
        raise
