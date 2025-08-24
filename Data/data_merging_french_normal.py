# File that merges the data to add the WERs to each line item. The Updated Data will be saved in "Updated Data" folder.
"""
Merge non-transformed French WERs into Acoustic Lines CSVs by line_id. This specific part of the data has its own script because the WER file is formatted differently than for the other categories. 

Expected layout:

.
├── Acoustic Lines/
│   ├── belgium_female_individual_sample_features.csv
│   └── ...
└── WERs/
    └── French Whisper v3/
        ├── belgium female.txt
        └── ...

Each WER .txt contains repeated blocks like:
    Line ID: 7457246003
    WER Score: 16.00
    Transcript: ...
    Reference: ...
    -------------------------------------------

Outputs go to:
- "Acoustic Lines (with WER)/<same_csv_name>"
"""

from __future__ import annotations
from pathlib import Path
import re
import sys
import json
import pandas as pd
from typing import Dict, Tuple

# -------------------------
# CONFIG – tweak if needed
# -------------------------
PROJECT_ROOT = "."
ACOUSTIC_DIR = "Data/Acoustic Lines"
WER_ROOT = "Data/WERs"
FRENCH_WER_SUBFOLDER = "French Whisper v3"
OUT_ACOUSTIC_DIR = "Data/Acoustic Lines (with WER)"
WER_COLUMN_NAME = "wer_score"

CSV_READ_KW = {"encoding": "utf-8"}
CSV_WRITE_KW = {"index": False, "encoding": "utf-8"}

# Regex to capture "Line ID: <digits>" followed (any lines later) by "WER Score: <num>"
WER_BLOCK_PATTERN = re.compile(
    r"Line ID:\s*(?P<line_id>\d+)\s*"
    r"(?:.+?\n)*?"
    r"WER Score:\s*(?P<wer>[-+]?\d+(?:\.\d+)?)",
    flags=re.IGNORECASE | re.DOTALL,
)

def parse_wer_file(path: Path) -> Dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    mapping: Dict[str, float] = {}
    for m in WER_BLOCK_PATTERN.finditer(text):
        lid = m.group("line_id").strip()
        try:
            wer = float(m.group("wer").strip())
        except ValueError:
            continue
        # Keep first occurrence if duplicates appear within the same file
        mapping.setdefault(lid, wer)
    return mapping

def parse_french_wers(wer_root: Path) -> Tuple[Dict[str, float], int]:
    base = wer_root / FRENCH_WER_SUBFOLDER
    if not base.exists():
        print(f"[ERROR] Missing French WER folder: {base}", file=sys.stderr)
        sys.exit(1)

    merged: Dict[str, float] = {}
    dup_across_files = 0

    for txt in sorted(base.rglob("*.txt")):
        file_map = parse_wer_file(txt)
        for lid, wer in file_map.items():
            if lid in merged:
                # duplicate line id across different WER files
                dup_across_files += 1
                continue
            merged[lid] = wer

    return merged, dup_across_files

def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, **CSV_READ_KW)
    if "line_id" not in df.columns:
        raise KeyError(f"'line_id' column not found in {csv_path}")
    df["line_id"] = df["line_id"].astype(str)
    return df

def merge_wer(df: pd.DataFrame, wer_map: Dict[str, float]) -> pd.DataFrame:
    df_out = df.copy()
    df_out[WER_COLUMN_NAME] = df_out["line_id"].map(wer_map)
    return df_out

def process_acoustic_folder(acoustic_dir: Path, out_dir: Path, wer_map: Dict[str, float]) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"files": 0, "rows_total": 0, "rows_with_wer": 0, "rows_missing_wer": 0, "written": 0}
    missing_wer_details = []

    for csv_path in sorted(acoustic_dir.rglob("*.csv")):
        stats["files"] += 1
        df = load_csv(csv_path)
        stats["rows_total"] += len(df)

        df_out = merge_wer(df, wer_map)
        with_ = int(df_out[WER_COLUMN_NAME].notna().sum())
        without_ = len(df_out) - with_
        stats["rows_with_wer"] += with_
        stats["rows_missing_wer"] += without_

        # Track which rows lack WER scores for this file
        if without_ > 0:
            missing_rows = df_out[df_out[WER_COLUMN_NAME].isna()]
            missing_wer_details.append({
                "filename": csv_path.name,
                "missing_count": without_,
                "missing_line_ids": missing_rows["line_id"].tolist()
            })

        rel = csv_path.relative_to(acoustic_dir)
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_path, **CSV_WRITE_KW)
        stats["written"] += 1

    stats["missing_wer_details"] = missing_wer_details
    return stats

def verify_specific_wer_scores(out_acoustic_dir: Path) -> bool:
    """Verify specific WER scores for validation."""
    errors = []
    
    # Check 1: france_female_individual_sample_features.csv, line ID 1942739731 should have WER 6.33
    france_female_path = out_acoustic_dir / "france_female_individual_sample_features.csv"
    if france_female_path.exists():
        try:
            df_france = pd.read_csv(france_female_path, **CSV_READ_KW)
            france_row = df_france[df_france["line_id"] == 1942739731]
            if len(france_row) == 0:
                errors.append(1)
            elif abs(france_row.iloc[0]["wer_score"] - 6.33) > 0.01:  # Allow small floating point differences
                errors.append(1)
        except Exception:
            errors.append(1)
    else:
        errors.append(1)
    
    # Check 2: ivory_coast_male_individual_sample_features.csv, line ID 2681516877 should have WER 2.17
    ivory_coast_male_path = out_acoustic_dir / "ivory_coast_male_individual_sample_features.csv"
    if ivory_coast_male_path.exists():
        try:
            df_ivory = pd.read_csv(ivory_coast_male_path, **CSV_READ_KW)
            ivory_row = df_ivory[df_ivory["line_id"] == 2681516877]
            if len(ivory_row) == 0:
                errors.append(2)
            elif abs(ivory_row.iloc[0]["wer_score"] - 2.17) > 0.01:  # Allow small floating point differences
                errors.append(2)
        except Exception:
            errors.append(2)
    else:
        errors.append(2)
    
    if errors:
        if len(errors) == 1:
            print(f"error: failed check {errors[0]}")
        else:
            print("error: failed check both")
        return False
    
    print("[VERIFICATION] All specific WER score checks passed.")
    return True

def verify_acoustic_data_integrity(acoustic_dir: Path, out_acoustic_dir: Path) -> bool:
    """Verify that acoustic features remain unchanged during the merge process."""
    print("[VERIFICATION] Checking acoustic data integrity...")
    
    # Columns to verify (excluding wer_score which is added during merge)
    columns_to_check = ["speaker_id", "rms_amplitude", "dc_offset", "articulation_rate", 
                        "mean_pitch", "pitch_std_dev", "hnr"]
    
    errors_found = False
    
    for csv_path in sorted(acoustic_dir.rglob("*.csv")):
        filename = csv_path.name
        rel_path = csv_path.relative_to(acoustic_dir)
        out_path = out_acoustic_dir / rel_path
        
        if not out_path.exists():
            print(f"error: output file missing for {filename}")
            errors_found = True
            continue
            
        try:
            # Load original and merged data
            df_original = pd.read_csv(csv_path, **CSV_READ_KW)
            df_merged = pd.read_csv(out_path, **CSV_READ_KW)
            
            # Ensure both dataframes have the same number of rows
            if len(df_original) != len(df_merged):
                print(f"error: row count mismatch in {filename} (original: {len(df_original)}, merged: {len(df_merged)})")
                errors_found = True
                continue
            
            # Check each row for data integrity
            for idx, (_, row_original) in enumerate(df_original.iterrows()):
                line_id = row_original["line_id"]
                
                # Find corresponding row in merged data
                merged_row = df_merged[df_merged["line_id"] == line_id]
                if len(merged_row) == 0:
                    print(f"error: line_id {line_id} missing in merged file {filename}")
                    errors_found = True
                    continue
                
                # Check each column for equality
                for col in columns_to_check:
                    if col not in row_original or col not in merged_row.iloc[0]:
                        print(f"error: column {col} missing in {filename} for line_id {line_id}")
                        errors_found = True
                        continue
                    
                    original_val = row_original[col]
                    merged_val = merged_row.iloc[0][col]
                    
                    # Handle floating point comparison with tolerance
                    if isinstance(original_val, (int, float)) and isinstance(merged_val, (int, float)):
                        if abs(original_val - merged_val) > 1e-10:  # Very small tolerance for floating point
                            print(f"error: data mismatch in {filename} line_id {line_id} column {col} (original: {original_val}, merged: {merged_val})")
                            errors_found = True
                    elif original_val != merged_val:
                        print(f"error: data mismatch in {filename} line_id {line_id} column {col} (original: {original_val}, merged: {merged_val})")
                        errors_found = True
                        
        except Exception as e:
            print(f"error: failed to verify {filename}: {str(e)}")
            errors_found = True
    
    if not errors_found:
        print("[VERIFICATION] All acoustic data integrity checks passed.")
        return True
    else:
        print("[VERIFICATION] Acoustic data integrity check failed.")
        return False

def verify_row_counts(acoustic_dir: Path, out_acoustic_dir: Path) -> bool:
    """Verify that each file has the same number of rows in both original and merged versions."""
    print("[VERIFICATION] Checking row count consistency...")
    
    errors_found = False
    
    for csv_path in sorted(acoustic_dir.rglob("*.csv")):
        filename = csv_path.name
        rel_path = csv_path.relative_to(acoustic_dir)
        out_path = out_acoustic_dir / rel_path
        
        if not out_path.exists():
            print(f"error: output file missing for {filename}")
            errors_found = True
            continue
            
        try:
            # Count rows in both files
            df_original = pd.read_csv(csv_path, **CSV_READ_KW)
            df_merged = pd.read_csv(out_path, **CSV_READ_KW)
            
            original_count = len(df_original)
            merged_count = len(df_merged)
            
            if original_count != merged_count:
                print(f"error: row count mismatch in {filename} (original: {original_count}, merged: {merged_count})")
                errors_found = True
            else:
                print(f"[OK] {filename}: {original_count} rows ✓")
                
        except Exception as e:
            print(f"error: failed to verify row counts for {filename}: {str(e)}")
            errors_found = True
    
    if not errors_found:
        print("[VERIFICATION] All row count checks passed.")
        return True
    else:
        print("[VERIFICATION] Row count verification failed.")
        return False

def verify_data_types_and_ranges(out_dir: Path) -> bool:
    """Verify data types and reasonable ranges for all columns."""
    print("[VERIFICATION] Checking data types and ranges...")
    
    errors_found = False
    
    for csv_path in sorted(out_dir.rglob("*.csv")):
        try:
            df = pd.read_csv(csv_path, **CSV_READ_KW)
            
            # Check WER scores are numeric and in reasonable range (0-200, allowing for very poor ASR performance)
            if "wer_score" in df.columns:
                wer_scores = df["wer_score"].dropna()
                if len(wer_scores) > 0:
                    if not pd.api.types.is_numeric_dtype(wer_scores):
                        print(f"error: WER scores in {csv_path.name} are not numeric")
                        errors_found = True
                    elif (wer_scores < 0).any():
                        print(f"error: WER scores in {csv_path.name} contain negative values")
                        errors_found = True
                    elif (wer_scores > 200).any():
                        print(f"warning: WER scores in {csv_path.name} exceed 200% - may indicate very poor ASR performance")
                        # Don't fail the check for this, just warn
            
            # Check acoustic features are numeric
            numeric_cols = ["rms_amplitude", "dc_offset", "articulation_rate", "mean_pitch", "pitch_std_dev", "hnr"]
            for col in numeric_cols:
                if col in df.columns:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        print(f"error: {col} in {csv_path.name} is not numeric")
                        errors_found = True
            
            # Check for extreme outliers (3+ standard deviations)
            for col in numeric_cols:
                if col in df.columns:
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        mean_val = col_data.mean()
                        std_val = col_data.std()
                        if std_val > 0:
                            outliers = col_data[abs(col_data - mean_val) > 3 * std_val]
                            if len(outliers) > 0:
                                print(f"warning: {len(outliers)} extreme outliers in {col} for {csv_path.name}")
                                
        except Exception as e:
            print(f"error: failed to verify data types for {csv_path.name}: {str(e)}")
            errors_found = True
    
    if not errors_found:
        print("[VERIFICATION] All data type and range checks passed.")
        return True
    else:
        print("[VERIFICATION] Data type and range checks failed.")
        return False

def verify_no_duplicates(out_dir: Path) -> bool:
    """Verify no duplicate line_ids within individual files."""
    print("[VERIFICATION] Checking for duplicate line_ids...")
    
    errors_found = False
    
    for csv_path in sorted(out_dir.rglob("*.csv")):
        try:
            df = pd.read_csv(csv_path, **CSV_READ_KW)
            
            # Check for duplicate line_ids within the same file
            duplicates = df[df.duplicated(subset=["line_id"], keep=False)]
            if len(duplicates) > 0:
                print(f"error: {len(duplicates)} duplicate line_ids found in {csv_path.name}")
                print(f"  Duplicate line_ids: {duplicates['line_id'].unique()}")
                errors_found = True
                
        except Exception as e:
            print(f"error: failed to check duplicates for {csv_path.name}: {str(e)}")
            errors_found = True
    
    if not errors_found:
        print("[VERIFICATION] No duplicate line_ids found.")
        return True
    else:
        print("[VERIFICATION] Duplicate detection failed.")
        return False

def main():
    root = Path(PROJECT_ROOT).resolve()
    acoustic_dir = (root / ACOUSTIC_DIR).resolve()
    wer_root = (root / WER_ROOT).resolve()
    out_acoustic_dir = (root / OUT_ACOUSTIC_DIR).resolve()

    if not acoustic_dir.exists():
        print(f"[ERROR] Missing folder: {acoustic_dir}", file=sys.stderr)
        sys.exit(1)
    if not wer_root.exists():
        print(f"[ERROR] Missing WER root: {wer_root}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Parsing non-transformed FRENCH WERs from '{FRENCH_WER_SUBFOLDER}' ...")
    french_map, dup_cross = parse_french_wers(wer_root)
    print(f"[INFO] Parsed {len(french_map):,} unique line_ids (duplicates across files: {dup_cross}).")

    print(f"[INFO] Merging into Acoustic Lines → {out_acoustic_dir} ...")
    stats = process_acoustic_folder(acoustic_dir, out_acoustic_dir, french_map)
    print(json.dumps({"acoustic_stats": stats}, indent=2))

    if stats["rows_missing_wer"]:
        print(f"\n[NOTE] {stats['rows_missing_wer']} rows lack French WERs — expected for non-French/Spanish files or lines absent in the French WER set.")
        print("\nDetailed breakdown of missing WER scores:")
        for detail in stats["missing_wer_details"]:
            print(f"  {detail['filename']}: {detail['missing_count']} missing WER scores")
            if detail['missing_count'] <= 10:  # Show line IDs if not too many
                print(f"    Missing line IDs: {detail['missing_line_ids']}")
            else:
                print(f"    Missing line IDs: {detail['missing_line_ids'][:5]}... and {detail['missing_count']-5} more")

    print("[INFO] Verifying specific WER scores...")
    specific_wer_check = verify_specific_wer_scores(out_acoustic_dir)

    print("[INFO] Verifying acoustic data integrity...")
    acoustic_integrity_check = verify_acoustic_data_integrity(acoustic_dir, out_acoustic_dir)

    print("[INFO] Verifying row counts...")
    row_count_check = verify_row_counts(acoustic_dir, out_acoustic_dir)

    print("\n[INFO] Running additional edge case checks...")
    data_type_check = verify_data_types_and_ranges(out_acoustic_dir)
    duplicate_check = verify_no_duplicates(out_acoustic_dir)

    # Summary of all checks
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    print(f"Specific WER score checks: {'✓ PASSED' if specific_wer_check else '✗ FAILED'}")
    print(f"Acoustic data integrity: {'✓ PASSED' if acoustic_integrity_check else '✗ FAILED'}")
    print(f"Row count verification: {'✓ PASSED' if row_count_check else '✗ FAILED'}")
    print(f"Data type and range checks: {'✓ PASSED' if data_type_check else '✗ FAILED'}")
    print(f"Duplicate detection: {'✓ PASSED' if duplicate_check else '✗ FAILED'}")
    
    all_passed = (specific_wer_check and acoustic_integrity_check and row_count_check and data_type_check and duplicate_check)
    
    if all_passed:
        print("\n🎉 ALL VERIFICATION CHECKS PASSED!")
    else:
        print("\n⚠️  SOME VERIFICATION CHECKS FAILED!")
    
    print("\n[DONE] Merge complete.")

if __name__ == "__main__":
    main()
