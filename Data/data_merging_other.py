# File that merges Spanish WERs and transformed acoustic data with WER scores.
# This script handles the different WER file format: <score> then line_id <id>
"""
Merge Spanish WERs into Acoustic Lines CSVs and merge transformed acoustic data with both French and Spanish WERs.

Expected layout:

.
├── Acoustic Lines/
│   ├── argentina_female_individual_sample_features.csv
│   └── ...
├── Transformed Acoustic Lines/
│   ├── argentina_female_individual_sample_features.csv
│   └── ...
└── WERs/
    ├── Spanish Whisper v3/
    │   ├── argentina female.txt
    │   └── ...
    ├── Transformed French Whisper v3/
    │   ├── transformed belgium female.txt
    │   └── ...
    └── Transformed Spanish Whisper v3/
        ├── transformed argentina female.txt
        └── ...

Each Spanish/Transformed WER .txt contains blocks like:
    0.0
    line_id 3886149120
    transcript norm: ...
    ref norm: ...
    -------------------------------------------

Outputs go to:
- "Acoustic Lines (with WER)/<same_csv_name>" (Spanish WERs added)
- "Transformed Acoustic Lines (with WER)/<same_csv_name>" (both French and Spanish WERs)
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
TRANSFORMED_ACOUSTIC_DIR = "Data/Transformed Acoustic Lines"
WER_ROOT = "Data/WERs"
SPANISH_WER_SUBFOLDER = "Spanish Whisper v3"
TRANSFORMED_FRENCH_WER_SUBFOLDER = "Transformed French Whisper v3"
TRANSFORMED_SPANISH_WER_SUBFOLDER = "Transformed Spanish Whisper v3"
OUT_ACOUSTIC_DIR = "Data/Acoustic Lines (with WER)"
OUT_TRANSFORMED_ACOUSTIC_DIR = "Data/Transformed Acoustic Lines (with WER)"
WER_COLUMN_NAME = "wer_score"

CSV_READ_KW = {"encoding": "utf-8"}
CSV_WRITE_KW = {"index": False, "encoding": "utf-8"}

# Regex to capture WER score followed by line_id for Spanish/Transformed format
SPANISH_WER_BLOCK_PATTERN = re.compile(
    r"(?P<wer>[-+]?\d+(?:\.\d+)?)\s*\n"
    r"line_id\s*(?P<line_id>\d+)",
    flags=re.IGNORECASE | re.DOTALL,
)

# Regex to capture "Line ID: <digits>" followed by "WER Score: <num>" for French format
FRENCH_WER_BLOCK_PATTERN = re.compile(
    r"Line ID:\s*(?P<line_id>\d+)\s*"
    r"(?:.+?\n)*?"
    r"WER Score:\s*(?P<wer>[-+]?\d+(?:\.\d+)?)",
    flags=re.IGNORECASE | re.DOTALL,
)

def parse_spanish_wer_file(path: Path) -> Dict[str, float]:
    """Parse Spanish/Transformed WER files with format: <score> then line_id <id>"""
    text = path.read_text(encoding="utf-8", errors="replace")
    mapping: Dict[str, float] = {}
    for m in SPANISH_WER_BLOCK_PATTERN.finditer(text):
        try:
            wer = float(m.group("wer").strip())
            lid = m.group("line_id").strip()
        except ValueError:
            continue
        # Keep first occurrence if duplicates appear within the same file
        mapping.setdefault(lid, wer)
    return mapping

def parse_french_wer_file(path: Path) -> Dict[str, float]:
    """Parse French WER files with format: Line ID: <id> then WER Score: <score>"""
    text = path.read_text(encoding="utf-8", errors="replace")
    mapping: Dict[str, float] = {}
    for m in FRENCH_WER_BLOCK_PATTERN.finditer(text):
        lid = m.group("line_id").strip()
        try:
            wer = float(m.group("wer").strip())
        except ValueError:
            continue
        # Keep first occurrence if duplicates appear within the same file
        mapping.setdefault(lid, wer)
    return mapping

def parse_spanish_wers(wer_root: Path) -> Tuple[Dict[str, float], int]:
    """Parse Spanish WER files and return mapping of line_id to WER score."""
    base = wer_root / SPANISH_WER_SUBFOLDER
    if not base.exists():
        print(f"[ERROR] Missing Spanish WER folder: {base}", file=sys.stderr)
        sys.exit(1)

    merged: Dict[str, float] = {}
    dup_across_files = 0

    for txt in sorted(base.rglob("*.txt")):
        file_map = parse_spanish_wer_file(txt)
        for lid, wer in file_map.items():
            if lid in merged:
                # duplicate line id across different WER files
                dup_across_files += 1
                continue
            merged[lid] = wer

    return merged, dup_across_files

def parse_transformed_wers(wer_root: Path) -> Tuple[Dict[str, float], Dict[str, float], int]:
    """Parse both transformed French and Spanish WER files."""
    french_base = wer_root / TRANSFORMED_FRENCH_WER_SUBFOLDER
    spanish_base = wer_root / TRANSFORMED_SPANISH_WER_SUBFOLDER
    
    if not french_base.exists():
        print(f"[ERROR] Missing Transformed French WER folder: {french_base}", file=sys.stderr)
        sys.exit(1)
    if not spanish_base.exists():
        print(f"[ERROR] Missing Transformed Spanish WER folder: {spanish_base}", file=sys.stderr)
        sys.exit(1)

    french_merged: Dict[str, float] = {}
    spanish_merged: Dict[str, float] = {}
    dup_across_files = 0

    # Parse French transformed WERs - they use Spanish format!
    for txt in sorted(french_base.rglob("*.txt")):
        file_map = parse_spanish_wer_file(txt)  # Use Spanish parser for French transformed files
        for lid, wer in file_map.items():
            if lid in french_merged:
                dup_across_files += 1
                continue
            french_merged[lid] = wer

    # Parse Spanish transformed WERs
    for txt in sorted(spanish_base.rglob("*.txt")):
        file_map = parse_spanish_wer_file(txt)
        for lid, wer in file_map.items():
            if lid in spanish_merged:
                dup_across_files += 1
                continue
            spanish_merged[lid] = wer

    return french_merged, spanish_merged, dup_across_files

def load_csv(csv_path: Path) -> pd.DataFrame:
    """Load CSV and ensure line_id column exists."""
    df = pd.read_csv(csv_path, **CSV_READ_KW)
    if "line_id" not in df.columns:
        raise KeyError(f"'line_id' column not found in {csv_path}")
    df["line_id"] = df["line_id"].astype(str)
    return df

def merge_wer(df: pd.DataFrame, wer_map: Dict[str, float]) -> pd.DataFrame:
    """Merge WER scores into dataframe."""
    df_out = df.copy()
    df_out[WER_COLUMN_NAME] = df_out["line_id"].map(wer_map)
    return df_out

def process_spanish_acoustic_folder(acoustic_dir: Path, out_dir: Path, spanish_wer_map: Dict[str, float]) -> dict:
    """Process Spanish-speaking countries and add Spanish WER scores."""
    print(f"[DEBUG] process_spanish_acoustic_folder called with {len(spanish_wer_map)} WER scores")
    print(f"[DEBUG] Processing files from: {acoustic_dir}")
    print(f"[DEBUG] Output directory: {out_dir}")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"files": 0, "rows_total": 0, "rows_with_wer": 0, "rows_missing_wer": 0, "written": 0}
    missing_wer_details = []

    # Spanish-speaking countries
    spanish_countries = ["argentina", "chile", "dominican_republic", "mexico", "spain"]
    
    for csv_path in sorted(acoustic_dir.rglob("*.csv")):
        filename = csv_path.name
        
        # Only process Spanish-speaking countries
        if not any(country in filename.lower() for country in spanish_countries):
            continue
            
        print(f"[DEBUG] Processing Spanish file: {filename}")
        stats["files"] += 1
        df = load_csv(csv_path)
        stats["rows_total"] += len(df)
        print(f"[DEBUG] Loaded {len(df)} rows from {filename}")

        df_out = merge_wer(df, spanish_wer_map)
        with_ = int(df_out[WER_COLUMN_NAME].notna().sum())
        without_ = len(df_out) - with_
        stats["rows_with_wer"] += with_
        stats["rows_missing_wer"] += without_
        print(f"[DEBUG] After merge: {with_} with WER, {without_} without WER")

        # Track which rows lack WER scores for this file
        if without_ > 0:
            missing_rows = df_out[df_out[WER_COLUMN_NAME].isna()]
            missing_wer_details.append({
                "filename": filename,
                "missing_count": without_,
                "missing_line_ids": missing_rows["line_id"].tolist()
            })

        rel = csv_path.relative_to(acoustic_dir)
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[DEBUG] Writing to: {out_path}")
        df_out.to_csv(out_path, **CSV_WRITE_KW)
        stats["written"] += 1
        print(f"[DEBUG] Successfully wrote {filename}")

    print(f"[DEBUG] process_spanish_acoustic_folder completed. Stats: {stats}")
    stats["missing_wer_details"] = missing_wer_details
    return stats

def process_transformed_acoustic_folder(acoustic_dir: Path, out_dir: Path, 
                                     french_wer_map: Dict[str, float], 
                                     spanish_wer_map: Dict[str, float]) -> dict:
    """Process transformed acoustic files and add appropriate WER scores."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"files": 0, "rows_total": 0, "rows_with_wer": 0, "rows_missing_wer": 0, "written": 0}
    missing_wer_details = []

    for csv_path in sorted(acoustic_dir.rglob("*.csv")):
        filename = csv_path.name
        stats["files"] += 1
        df = load_csv(csv_path)
        stats["rows_total"] += len(df)

        # Determine which WER map to use based on country
        if any(country in filename.lower() for country in ["argentina", "chile", "dominican_republic", "mexico", "spain"]):
            wer_map = spanish_wer_map
            country_type = "Spanish"
        else:
            wer_map = french_wer_map
            country_type = "French"

        df_out = merge_wer(df, wer_map)
        with_ = int(df_out[WER_COLUMN_NAME].notna().sum())
        without_ = len(df_out) - with_
        stats["rows_with_wer"] += with_
        stats["rows_missing_wer"] += without_

        # Track which rows lack WER scores for this file
        if without_ > 0:
            missing_rows = df_out[df_out[WER_COLUMN_NAME].isna()]
            missing_wer_details.append({
                "filename": filename,
                "missing_count": without_,
                "missing_line_ids": missing_rows["line_id"].tolist(),
                "country_type": country_type
            })

        rel = csv_path.relative_to(acoustic_dir)
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Reorder columns to match transformed format: rms_amplitude,dc_offset,articulation_rate,mean_pitch,pitch_std_dev,hnr,original_speaker_id,line_id,wer_score
        column_order = ["rms_amplitude", "dc_offset", "articulation_rate", "mean_pitch", "pitch_std_dev", "hnr", "original_speaker_id", "line_id", "wer_score"]
        df_out = df_out[column_order]
        
        df_out.to_csv(out_path, **CSV_WRITE_KW)
        stats["written"] += 1

    stats["missing_wer_details"] = missing_wer_details
    return stats

def verify_specific_wer_scores_spanish(out_acoustic_dir: Path) -> bool:
    """Verify specific WER scores for Spanish files."""
    print("[VERIFICATION] Checking specific Spanish WER scores...")
    
    # Specific WER score checks for Spanish non-transformed files
    spanish_checks = [
        {"line_id": 8197521612, "expected_wer": 1.5151515151515151, "description": "Spanish non-transformed"},
        {"line_id": 4507228429, "expected_wer": 0.0, "description": "Spanish non-transformed"},
        {"line_id": 5113853266, "expected_wer": 2.4096385542168677, "description": "Spanish non-transformed"}
    ]
    
    errors = []
    
    for check in spanish_checks:
        line_id = check["line_id"]
        expected_wer = check["expected_wer"]
        description = check["description"]
        
        # Search for this line_id across all Spanish files
        found = False
        for csv_path in sorted(out_acoustic_dir.rglob("*.csv")):
            filename = csv_path.name
            
            # Only check Spanish-speaking countries
            if not any(country in filename.lower() for country in ["argentina", "chile", "dominican_republic", "mexico", "spain"]):
                continue
                
            try:
                df = pd.read_csv(csv_path, **CSV_READ_KW)
                row = df[df["line_id"] == line_id]
                
                if len(row) > 0:
                    found = True
                    actual_wer = row.iloc[0]["wer_score"]
                    
                    if pd.isna(actual_wer):
                        errors.append(f"Spanish {description} - line_id {line_id}: WER score is NaN in {filename}")
                    elif abs(actual_wer - expected_wer) > 0.01:  # Allow small floating point differences
                        errors.append(f"Spanish {description} - line_id {line_id}: expected {expected_wer}, got {actual_wer} in {filename}")
                    else:
                        print(f"[OK] Spanish {description} - line_id {line_id}: {actual_wer} ✓")
                    break
                    
            except Exception as e:
                errors.append(f"Spanish {description} - line_id {line_id}: error reading {filename}: {str(e)}")
                break
        
        if not found:
            errors.append(f"Spanish {description} - line_id {line_id}: not found in any Spanish file")
    
    if errors:
        print("[VERIFICATION] Spanish WER score checks failed:")
        for error in errors:
            print(f"  error: {error}")
        return False
    
    print("[VERIFICATION] All Spanish WER score checks passed.")
    return True

def verify_specific_wer_scores_transformed(out_transformed_dir: Path) -> bool:
    """Verify specific WER scores for transformed files."""
    print("[VERIFICATION] Checking specific transformed WER scores...")
    
    # Specific WER score checks for transformed files
    transformed_checks = [
        # French transformed
        {"line_id": 4765924089, "expected_wer": 22.413793103448278, "description": "French transformed"},
        {"line_id": 3711871937, "expected_wer": 21.818181818181817, "description": "French transformed"},
        {"line_id": 7487436958, "expected_wer": 12.903225806451612, "description": "French transformed"},
        # Spanish transformed
        {"line_id": 6650258635, "expected_wer": 11.11111111111111, "description": "Spanish transformed"},
        {"line_id": 6650256992, "expected_wer": 6.976744186046512, "description": "Spanish transformed"},
        {"line_id": 2589222884, "expected_wer": 12.5, "description": "Spanish transformed"}
    ]
    
    errors = []
    
    for check in transformed_checks:
        line_id = check["line_id"]
        expected_wer = check["expected_wer"]
        description = check["description"]
        
        # Search for this line_id across all transformed files
        found = False
        for csv_path in sorted(out_transformed_dir.rglob("*.csv")):
            filename = csv_path.name
            
            try:
                df = pd.read_csv(csv_path, **CSV_READ_KW)
                row = df[df["line_id"] == line_id]
                
                if len(row) > 0:
                    found = True
                    actual_wer = row.iloc[0]["wer_score"]
                    
                    if pd.isna(actual_wer):
                        errors.append(f"{description} - line_id {line_id}: WER score is NaN in {filename}")
                    elif abs(actual_wer - expected_wer) > 0.01:  # Allow small floating point differences
                        errors.append(f"{description} - line_id {line_id}: expected {expected_wer}, got {actual_wer} in {filename}")
                    else:
                        print(f"[OK] {description} - line_id {line_id}: {actual_wer} ✓")
                    break
                    
            except Exception as e:
                errors.append(f"{description} - line_id {line_id}: error reading {filename}: {str(e)}")
                break
        
        if not found:
            errors.append(f"{description} - line_id {line_id}: not found in any transformed file")
    
    if errors:
        print("[VERIFICATION] Transformed WER score checks failed:")
        for error in errors:
            print(f"  error: {error}")
        return False
    
    print("[VERIFICATION] All transformed WER score checks passed.")
    return True

def verify_acoustic_data_integrity(acoustic_dir: Path, out_acoustic_dir: Path, is_transformed: bool = False) -> bool:
    """Verify that acoustic features remain unchanged during the merge process."""
    print(f"[VERIFICATION] Checking {'transformed ' if is_transformed else ''}acoustic data integrity...")
    
    # Columns to verify based on file type
    if is_transformed:
        # Transformed files: rms_amplitude,dc_offset,articulation_rate,mean_pitch,pitch_std_dev,hnr,original_speaker_id,line_id
        columns_to_check = ["rms_amplitude", "dc_offset", "articulation_rate", "mean_pitch", "pitch_std_dev", "hnr", "original_speaker_id"]
        speaker_id_col = "original_speaker_id"
    else:
        # Regular acoustic files: speaker_id,rms_amplitude,dc_offset,articulation_rate,mean_pitch,pitch_std_dev,hnr,line_id
        columns_to_check = ["speaker_id", "rms_amplitude", "dc_offset", "articulation_rate", "mean_pitch", "pitch_std_dev", "hnr"]
        speaker_id_col = "speaker_id"
    
    errors_found = False
    missing_speaker_ids = {}
    
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
            
            # Check for missing speaker_id values
            if speaker_id_col in df_original.columns:
                missing_speaker_rows = df_original[df_original[speaker_id_col].isna()]
                if len(missing_speaker_rows) > 0:
                    missing_speaker_ids[filename] = {
                        "count": len(missing_speaker_rows),
                        "line_ids": missing_speaker_rows["line_id"].tolist()
                    }
            
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
    
    # Report missing speaker_id values
    if missing_speaker_ids:
        print(f"\n[INFO] Files with missing {speaker_id_col} values:")
        for filename, details in missing_speaker_ids.items():
            print(f"  {filename}: {details['count']} missing {speaker_id_col} values")
            if details['count'] <= 10:
                print(f"    Missing line IDs: {details['line_ids']}")
            else:
                print(f"    Missing line IDs: {details['line_ids'][:5]}... and {details['count']-5} more")
    
    if not errors_found:
        print(f"[VERIFICATION] All {'transformed ' if is_transformed else ''}acoustic data integrity checks passed.")
        return True
    else:
        print(f"[VERIFICATION] {'Transformed ' if is_transformed else ''}acoustic data integrity check failed.")
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
    print("[DEBUG] Main function started!")
    print("[DEBUG] Current working directory:", Path.cwd())
    
    root = Path(PROJECT_ROOT).resolve()
    acoustic_dir = (root / ACOUSTIC_DIR).resolve()
    transformed_acoustic_dir = (root / TRANSFORMED_ACOUSTIC_DIR).resolve()
    wer_root = (root / WER_ROOT).resolve()
    out_acoustic_dir = (root / OUT_ACOUSTIC_DIR).resolve()
    out_transformed_acoustic_dir = (root / OUT_TRANSFORMED_ACOUSTIC_DIR).resolve()

    print("[DEBUG] Paths resolved:")
    print(f"  root: {root}")
    print(f"  acoustic_dir: {acoustic_dir}")
    print(f"  wer_root: {wer_root}")
    print(f"  out_acoustic_dir: {out_acoustic_dir}")

    # Check if required directories exist
    if not acoustic_dir.exists():
        print(f"[ERROR] Missing folder: {acoustic_dir}", file=sys.stderr)
        sys.exit(1)
    if not transformed_acoustic_dir.exists():
        print(f"[ERROR] Missing folder: {transformed_acoustic_dir}", file=sys.stderr)
        sys.exit(1)
    if not wer_root.exists():
        print(f"[ERROR] Missing WER root: {wer_root}", file=sys.stderr)
        sys.exit(1)

    # Step 1: Parse Spanish WERs for Acoustic Lines
    print(f"[INFO] Parsing Spanish WERs from '{SPANISH_WER_SUBFOLDER}' ...")
    spanish_map, spanish_dup_cross = parse_spanish_wers(wer_root)
    print(f"[INFO] Parsed {len(spanish_map):,} unique Spanish line_ids (duplicates across files: {spanish_dup_cross}).")
    print(f"[DEBUG] Spanish map size: {len(spanish_map)}")

    # Step 2: Parse Transformed WERs (both French and Spanish)
    print(f"[INFO] Parsing Transformed WERs from both '{TRANSFORMED_FRENCH_WER_SUBFOLDER}' and '{TRANSFORMED_SPANISH_WER_SUBFOLDER}' ...")
    print(f"[NOTE] Both folders use the same format: <score> then line_id <id>")
    transformed_french_map, transformed_spanish_map, transformed_dup_cross = parse_transformed_wers(wer_root)
    print(f"[INFO] Parsed {len(transformed_french_map):,} unique transformed French line_ids and {len(transformed_spanish_map):,} unique transformed Spanish line_ids (duplicates across files: {transformed_dup_cross}).")

    # Step 3: Merge Spanish WERs into Acoustic Lines (Spanish-speaking countries only)
    print(f"[INFO] Merging Spanish WERs into Acoustic Lines → {out_acoustic_dir} ...")
    print(f"[DEBUG] About to call process_spanish_acoustic_folder with {len(spanish_map)} WER scores")
    spanish_stats = process_spanish_acoustic_folder(acoustic_dir, out_acoustic_dir, spanish_map)
    print(json.dumps({"spanish_acoustic_stats": spanish_stats}, indent=2))
    print(f"[DEBUG] Spanish processing completed with stats: {spanish_stats}")

    if spanish_stats["rows_missing_wer"]:
        print(f"\n[NOTE] {spanish_stats['rows_missing_wer']} Spanish rows lack WER scores.")
        print("\nDetailed breakdown of missing Spanish WER scores:")
        for detail in spanish_stats["missing_wer_details"]:
            print(f"  {detail['filename']}: {detail['missing_count']} missing WER scores")
            if detail['missing_count'] <= 10:  # Show line IDs if not too many
                print(f"    Missing line IDs: {detail['missing_line_ids']}")
            else:
                print(f"    Missing line IDs: {detail['missing_line_ids'][:5]}... and {detail['missing_count']-5} more")

    # Step 4: Merge Transformed WERs into Transformed Acoustic Lines
    print(f"[INFO] Merging Transformed WERs into Transformed Acoustic Lines → {out_transformed_acoustic_dir} ...")
    transformed_stats = process_transformed_acoustic_folder(transformed_acoustic_dir, out_transformed_acoustic_dir, 
                                                         transformed_french_map, transformed_spanish_map)
    print(json.dumps({"transformed_acoustic_stats": transformed_stats}, indent=2))

    if transformed_stats["rows_missing_wer"]:
        print(f"\n[NOTE] {transformed_stats['rows_missing_wer']} transformed rows lack WER scores.")
        print("\nDetailed breakdown of missing transformed WER scores:")
        for detail in transformed_stats["missing_wer_details"]:
            print(f"  {detail['filename']}: {detail['missing_count']} missing WER scores ({detail['country_type']})")
            if detail['missing_count'] <= 10:  # Show line IDs if not too many
                print(f"    Missing line IDs: {detail['missing_line_ids']}")
            else:
                print(f"    Missing line IDs: {detail['missing_line_ids'][:5]}... and {detail['missing_count']-5} more")

    # Step 5: Run all verification checks
    print("\n" + "="*60)
    print("VERIFICATION CHECKS")
    print("="*60)
    
    print("\n[INFO] Verifying specific WER scores for Spanish files...")
    spanish_wer_check = verify_specific_wer_scores_spanish(out_acoustic_dir)

    print("\n[INFO] Verifying specific WER scores for transformed files...")
    transformed_wer_check = verify_specific_wer_scores_transformed(out_transformed_acoustic_dir)

    print("\n[INFO] Verifying acoustic data integrity...")
    acoustic_integrity_check = verify_acoustic_data_integrity(acoustic_dir, out_acoustic_dir, is_transformed=False)

    print("\n[INFO] Verifying transformed acoustic data integrity...")
    transformed_integrity_check = verify_acoustic_data_integrity(transformed_acoustic_dir, out_transformed_acoustic_dir, is_transformed=True)

    print("\n[INFO] Verifying row counts...")
    row_count_check = verify_row_counts(acoustic_dir, out_acoustic_dir)

    print("\n[INFO] Verifying transformed row counts...")
    transformed_row_count_check = verify_row_counts(transformed_acoustic_dir, out_transformed_acoustic_dir)

    print("\n[INFO] Running additional edge case checks...")
    data_type_check = verify_data_types_and_ranges(out_acoustic_dir)
    duplicate_check = verify_no_duplicates(out_acoustic_dir)

    # Summary of all checks
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    print(f"Spanish WER checks: {'✓ PASSED' if spanish_wer_check else '✗ FAILED'}")
    print(f"Transformed WER checks: {'✓ PASSED' if transformed_wer_check else '✗ FAILED'}")
    print(f"Acoustic data integrity: {'✓ PASSED' if acoustic_integrity_check else '✗ FAILED'}")
    print(f"Transformed data integrity: {'✓ PASSED' if transformed_integrity_check else '✗ FAILED'}")
    print(f"Row count verification: {'✓ PASSED' if row_count_check else '✗ FAILED'}")
    print(f"Transformed row count verification: {'✓ PASSED' if transformed_row_count_check else '✗ FAILED'}")
    print(f"Data type and range checks: {'✓ PASSED' if data_type_check else '✗ FAILED'}")
    print(f"Duplicate detection: {'✓ PASSED' if duplicate_check else '✗ FAILED'}")
    
    all_passed = (spanish_wer_check and transformed_wer_check and 
                  acoustic_integrity_check and transformed_integrity_check and
                  row_count_check and transformed_row_count_check and
                  data_type_check and duplicate_check)
    
    if all_passed:
        print("\n🎉 ALL VERIFICATION CHECKS PASSED!")
    else:
        print("\n⚠️  SOME VERIFICATION CHECKS FAILED!")
    
    print("\n[DONE] All merges and verifications complete.")

if __name__ == "__main__":
    main() 