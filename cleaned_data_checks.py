#!/usr/bin/env python3
"""
Script to count total lines, missing values, and check line ID mapping in cleaned datasets.
"""

import pandas as pd
from pathlib import Path

def get_available_folders():
    """Get list of available folders in the Data directory."""
    data_dir = Path("Data")
    if not data_dir.exists():
        return []
    
    folders = [f.name for f in data_dir.iterdir() if f.is_dir()]
    return sorted(folders)

def display_available_folders():
    """Display available folders for user selection."""
    folders = get_available_folders()
    
    print("📁 AVAILABLE FOLDERS IN Data/ DIRECTORY:")
    print("=" * 50)
    
    for i, folder in enumerate(folders, 1):
        print(f"{i:2d}. {folder}")
    
    print()
    return folders

def get_user_selection():
    """Get folder selection from user."""
    folders = get_available_folders()
    
    if not folders:
        print("❌ No folders found in Data/ directory")
        return []
    
    print("🔍 SELECT FOLDERS TO ANALYZE (1-4 folders)")
    print("Enter folder numbers separated by spaces (e.g., '1 3 4')")
    print("Or enter 'all' to analyze all folders")
    print()
    
    while True:
        try:
            user_input = input("Enter your selection: ").strip()
            
            if user_input.lower() == 'all':
                return folders
            
            # Parse user input
            selected_indices = [int(x) - 1 for x in user_input.split()]
            
            # Validate input
            if len(selected_indices) < 1 or len(selected_indices) > 4:
                print("❌ Please select 1 to 4 folders")
                continue
            
            if any(i < 0 or i >= len(folders) for i in selected_indices):
                print("❌ Invalid folder number(s). Please use numbers from the list above.")
                continue
            
            selected_folders = [folders[i] for i in selected_indices]
            return selected_folders
            
        except ValueError:
            print("❌ Invalid input. Please enter numbers separated by spaces.")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            return []

def check_speaker_id_mapping():
    """Check mapping of speaker_id to original_speaker_id between cleaned folders."""
    
    print("\n🔍 CHECKING SPEAKER ID MAPPING BETWEEN CLEANED FOLDERS")
    print("=" * 70)
    
    # Define the two cleaned folders to compare
    folder1 = "Acoustic Lines (with WER), Cleaned"
    folder2 = "Transformed Acoustic Lines (with WER), Cleaned"
    
    folder1_path = Path("Data") / folder1
    folder2_path = Path("Data") / folder2
    
    if not folder1_path.exists() or not folder2_path.exists():
        print("❌ One or both cleaned folders not found")
        return
    
    print(f"📁 Comparing: {folder1}")
    print(f"📁 With: {folder2}")
    print("-" * 70)
    
    # Get all CSV files from both folders
    csv_files1 = list(folder1_path.glob("*.csv"))
    csv_files2 = list(folder2_path.glob("*.csv"))
    
    if not csv_files1 or not csv_files2:
        print("❌ No CSV files found in one or both folders")
        return
    
    # Create a mapping of filename to file path for both folders
    files1_dict = {f.name: f for f in csv_files1}
    files2_dict = {f.name: f for f in csv_files2}
    
    # Find common files (same country/gender pairs)
    common_files = set(files1_dict.keys()) & set(files2_dict.keys())
    
    if not common_files:
        print("❌ No matching files found between the two folders")
        return
    
    print(f"✅ Found {len(common_files)} matching country/gender pair files")
    print()
    
    total_issues = 0
    
    for filename in sorted(common_files):
        print(f"🔍 Checking: {filename}")
        
        try:
            # Read both files
            df1 = pd.read_csv(files1_dict[filename])
            df2 = pd.read_csv(files2_dict[filename])
            
            # Check if required columns exist in both
            if 'speaker_id' not in df1.columns:
                print(f"  ❌ Missing 'speaker_id' column in {folder1}")
                total_issues += 1
                continue
                
            if 'original_speaker_id' not in df2.columns:
                print(f"  ❌ Missing 'original_speaker_id' column in {folder2}")
                total_issues += 1
                continue
            
            # Get speaker IDs from both files
            speaker_ids1 = set(df1['speaker_id'].astype(str))
            original_speaker_ids2 = set(df2['original_speaker_id'].astype(str))
            
            # Check for mapping
            only_in_1 = speaker_ids1 - original_speaker_ids2
            only_in_2 = original_speaker_ids2 - speaker_ids1
            common_ids = speaker_ids1 & original_speaker_ids2
            
            # Calculate counts
            total_in_1 = len(speaker_ids1)
            total_in_2 = len(original_speaker_ids2)
            matched_count = len(common_ids)
            missing_from_2 = len(only_in_1)
            missing_from_1 = len(only_in_2)
            
            issues_found = False
            
            if only_in_1:
                print(f"  ❌ Speaker IDs only in {folder1}: {missing_from_2} IDs")
                print(f"      Examples: {list(only_in_1)[:5]}")
                issues_found = True
                total_issues += 1
            
            if only_in_2:
                print(f"  ❌ Original Speaker IDs only in {folder2}: {missing_from_1} IDs")
                print(f"      Examples: {list(only_in_2)[:5]}")
                issues_found = True
                total_issues += 1
            
            # Print summary counts
            print(f"  📊 Summary:")
            print(f"      {folder1} (speaker_id): {total_in_1} total speaker IDs")
            print(f"      {folder2} (original_speaker_id): {total_in_2} total speaker IDs")
            print(f"      ✅ Matched: {matched_count} speaker IDs")
            print(f"      ❌ Missing from {folder2}: {missing_from_2} speaker IDs")
            if missing_from_1 > 0:
                print(f"      ❌ Missing from {folder1}: {missing_from_1} speaker IDs")
            
            print()
            
        except Exception as e:
            print(f"  ❌ Error processing {filename}: {e}")
            total_issues += 1
            print()
    
    # Summary
    print("=" * 70)
    print("🎯 SPEAKER ID MAPPING SUMMARY")
    print("=" * 70)
    
    if total_issues == 0:
        print("✅ All country/gender pair files have perfect speaker ID mapping!")
    else:
        print(f"⚠️  Found {total_issues} files with mapping issues")
        print("   Check the details above for specific problems")
        
        # Calculate overall statistics
        total_original = sum(len(pd.read_csv(files1_dict[fname])['speaker_id'].unique()) for fname in common_files)
        total_transformed = sum(len(pd.read_csv(files2_dict[fname])['original_speaker_id'].unique()) for fname in common_files)
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Total unique speaker IDs in {folder1}: {total_original:,}")
        print(f"   Total unique original speaker IDs in {folder2}: {total_transformed:,}")
        print(f"   Total missing from {folder2}: {total_original - total_transformed:,}")
        print(f"   Data retention rate: {(total_transformed / total_original * 100):.1f}%")
    
    return total_issues

def analyze_selected_folders(selected_folders):
    """Analyze the selected folders for line counts and missing values."""
    
    total_lines = 0
    total_missing = 0
    
    print(f"\n🧹 ANALYZING SELECTED FOLDERS ({len(selected_folders)} folders)")
    print("=" * 60)
    
    for folder_name in selected_folders:
        folder_path = Path("Data") / folder_name
        
        if not folder_path.exists():
            print(f"❌ Folder not found: {folder_path}")
            continue
            
        print(f"\n📁 Analyzing: {folder_name}")
        print("-" * 50)
        
        # Get all CSV files
        csv_files = list(folder_path.glob("*.csv"))
        
        if not csv_files:
            print("  ⚠️  No CSV files found in this folder")
            continue
        
        folder_lines = 0
        folder_missing = 0
        
        for csv_file in csv_files:
            try:
                # Read the data
                df = pd.read_csv(csv_file)
                
                # Count lines
                lines = len(df)
                folder_lines += lines
                
                # Count missing values
                missing = df.isnull().sum().sum()
                folder_missing += missing
                
                print(f"  {csv_file.name}: {lines:,} lines, {missing} missing values")
                
                # If there are missing values, show details
                if missing > 0:
                    missing_by_col = df.isnull().sum()
                    for col, missing_count in missing_by_col.items():
                        if missing_count > 0:
                            print(f"    • {col}: {missing_count} missing")
                
            except Exception as e:
                print(f"  ❌ Error reading {csv_file.name}: {e}")
        
        total_lines += folder_lines
        total_missing += folder_missing
        
        print(f"\n  📊 {folder_name} Summary:")
        print(f"    Total lines: {folder_lines:,}")
        print(f"    Total missing values: {folder_missing}")
        print(f"    Files processed: {len(csv_files)}")
    
    print("\n" + "=" * 60)
    print("🎯 OVERALL SUMMARY")
    print("=" * 60)
    print(f"Total lines across selected folders: {total_lines:,}")
    print(f"Total missing values across selected folders: {total_missing}")
    
    if total_missing == 0:
        print("✅ All selected datasets have NO missing values!")
    else:
        print(f"⚠️  {total_missing} missing values present in selected datasets")
    
    return total_lines, total_missing

def main():
    """Main function to run the analysis."""
    print("🚀 CLEANED DATA CHECKS AND ANALYSIS")
    print("=" * 60)
    
    # First, check speaker ID mapping between cleaned folders
    check_speaker_id_mapping()
    
    print("\n" + "=" * 60)
    
    # Display available folders for additional analysis
    available_folders = display_available_folders()
    
    if not available_folders:
        return
    
    # Get user selection
    selected_folders = get_user_selection()
    
    if not selected_folders:
        return
    
    print(f"\n✅ Selected folders: {', '.join(selected_folders)}")
    
    # Analyze selected folders
    analyze_selected_folders(selected_folders)

if __name__ == "__main__":
    main() 