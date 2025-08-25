#!/usr/bin/env python3
"""
Data Cleaning Analysis Script for ASR Analysis Project

This script analyzes all four data folders to:
1. Detect missing values and their locations
2. Detect outliers and their locations  
3. Compare column consistency across folders
4. Provide additional cleaning recommendations for regression

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

class DataCleaningAnalyzer:
    def __init__(self, data_root="Data"):
        """
        Initialize the analyzer with the data root directory.
        
        Args:
            data_root (str): Path to the data directory
        """
        self.data_root = Path(data_root)
        self.folders = [
            "Acoustic Lines",
            "Acoustic Lines (with WER)", 
            "Transformed Acoustic Lines",
            "Transformed Acoustic Lines (with WER)"
        ]
        self.results = {}
        
    def get_all_csv_files(self):
        """Get all CSV files from the four data folders."""
        all_files = {}
        for folder in self.folders:
            folder_path = self.data_root / folder
            if folder_path.exists():
                csv_files = list(folder_path.glob("*.csv"))
                all_files[folder] = csv_files
            else:
                print(f"Warning: Folder {folder} not found")
                all_files[folder] = []
        return all_files
    
    def analyze_missing_values(self, df, file_name):
        """
        Analyze missing values in a dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            file_name (str): Name of the file being analyzed
            
        Returns:
            dict: Missing value analysis results
        """
        missing_info = {
            'total_missing': df.isnull().sum().sum(),
            'missing_by_column': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'rows_with_missing': df[df.isnull().any(axis=1)].shape[0],
            'missing_locations': {}
        }
        
        # Find specific locations of missing values
        for col in df.columns:
            missing_indices = df[df[col].isnull()].index.tolist()
            if missing_indices:
                missing_info['missing_locations'][col] = missing_indices[:10]  # Limit to first 10
                
        return missing_info
    
    def analyze_missing_values_demographics(self, df, file_name):
        """
        Analyze missing values by demographic factors (gender, country).
        
        Args:
            df (pd.DataFrame): Input dataframe
            file_name (str): Name of the file being analyzed
            
        Returns:
            dict: Demographic analysis of missing values
        """
        # Extract demographic information from filename
        # Expected format: country_gender_individual_sample_features.csv
        filename_parts = file_name.replace('_individual_sample_features.csv', '').split('_')
        
        if len(filename_parts) >= 2:
            country = filename_parts[0]
            gender = filename_parts[1]
        else:
            country = "unknown"
            gender = "unknown"
        
        # Find missing values
        missing_mask = df.isnull().any(axis=1)
        missing_rows = df[missing_mask]
        
        demographics = {
            'file_name': file_name,
            'country': country,
            'gender': gender,
            'total_rows': len(df),
            'missing_rows': len(missing_rows),
            'missing_percentage': (len(missing_rows) / len(df)) * 100,
            'missing_by_column': {}
        }
        
        # Analyze missing values by column
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                demographics['missing_by_column'][col] = {
                    'count': missing_count,
                    'percentage': (missing_count / len(df)) * 100
                }
        
        return demographics
    
    def detect_outliers(self, df, file_name, method='zscore'):
        """
        Detect outliers using z-score method (more than 3 std from mean).
        
        Args:
            df (pd.DataFrame): Input dataframe
            file_name (str): Name of the file being analyzed
            method (str): Method for outlier detection (default: 'zscore')
            
        Returns:
            dict: Outlier analysis results
        """
        outlier_info = {
            'total_outliers': 0,
            'outliers_by_column': {},
            'outlier_locations': {},
            'outlier_percentage': {},
            'top_outliers': []  # Store top outliers with their details
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Calculate mean and std for this column in this file
            col_mean = df[col].mean()
            col_std = df[col].std()
            
            # Define outliers as more than 3 standard deviations from the mean
            lower_bound = col_mean - 3 * col_std
            upper_bound = col_mean + 3 * col_std
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            outlier_count = len(outliers)
            outlier_info['outliers_by_column'][col] = outlier_count
            outlier_info['total_outliers'] += outlier_count
            
            if outlier_count > 0:
                outlier_info['outlier_locations'][col] = outliers.index.tolist()[:10]  # Limit to first 10
                outlier_info['outlier_percentage'][col] = (outlier_count / len(df)) * 100
                
                # Store outlier details for ranking
                for idx in outliers.index:
                    outlier_value = df.loc[idx, col]
                    # Calculate how many standard deviations away from mean
                    z_score = abs((outlier_value - col_mean) / col_std)
                    outlier_info['top_outliers'].append({
                        'file': file_name,
                        'column': col,
                        'row_index': idx,
                        'value': outlier_value,
                        'mean': col_mean,
                        'std': col_std,
                        'z_score': z_score,
                        'deviation': abs(outlier_value - col_mean)
                    })
            else:
                outlier_info['outlier_percentage'][col] = 0.0
                
        return outlier_info
    
    def analyze_column_consistency(self):
        """
        Analyze column consistency across all folders.
        
        Returns:
            dict: Column consistency analysis results
        """
        all_files = self.get_all_csv_files()
        column_analysis = {}
        
        for folder, files in all_files.items():
            if not files:
                continue
                
            # Read first file from each folder to get column structure
            first_file = files[0]
            try:
                df = pd.read_csv(first_file)
                column_analysis[folder] = {
                    'columns': list(df.columns),
                    'column_count': len(df.columns),
                    'sample_file': first_file.name
                }
            except Exception as e:
                print(f"Error reading {first_file}: {e}")
                column_analysis[folder] = {'error': str(e)}
        
        # Compare columns across folders
        all_columns = set()
        for folder, info in column_analysis.items():
            if 'columns' in info:
                all_columns.update(info['columns'])
        
        column_analysis['consistency_check'] = {
            'all_unique_columns': sorted(list(all_columns)),
            'total_unique_columns': len(all_columns)
        }
        
        return column_analysis
    
    def analyze_data_types_and_ranges(self, df, file_name):
        """
        Analyze data types and value ranges for regression preparation.
        
        Args:
            df (pd.DataFrame): Input dataframe
            file_name (str): Name of the file being analyzed
            
        Returns:
            dict: Data type and range analysis
        """
        analysis = {
            'data_types': df.dtypes.to_dict(),
            'numeric_columns': [],
            'categorical_columns': [],
            'value_ranges': {},
            'unique_counts': {}
        }
        
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                analysis['numeric_columns'].append(col)
                analysis['value_ranges'][col] = {
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean(),
                    'std': df[col].std()
                }
            else:
                analysis['categorical_columns'].append(col)
                analysis['unique_counts'][col] = df[col].nunique()
                
        return analysis
    
    def create_outlier_histograms(self, results, output_dir="outlier_plots"):
        """
        Create histograms showing outliers compared to the distribution of values.
        
        Args:
            results (dict): Analysis results containing outlier information
            output_dir (str): Directory to save the plots
        """
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Group outliers by file and column
        file_column_outliers = {}
        
        for outlier in results['all_outliers']:
            file_col_key = (outlier['file'], outlier['column'])
            if file_col_key not in file_column_outliers:
                file_column_outliers[file_col_key] = []
            file_column_outliers[file_col_key].append(outlier)
        
        # Create plots for each file-column combination with outliers
        for (file_name, column), outliers in file_column_outliers.items():
            try:
                # Find the file path
                file_path = None
                for folder in self.folders:
                    folder_path = self.data_root / folder
                    if folder_path.exists():
                        potential_file = folder_path / file_name
                        if potential_file.exists():
                            file_path = potential_file
                            break
                
                if file_path is None:
                    continue
                
                # Read the data
                df = pd.read_csv(file_path)
                
                if column not in df.columns:
                    continue
                
                # Create the plot
                plt.figure(figsize=(12, 8))
                
                # Plot histogram of all values
                plt.hist(df[column].dropna(), bins=50, alpha=0.7, color='skyblue', 
                        label='All values', density=True)
                
                # Plot outliers
                outlier_values = [o['value'] for o in outliers]
                plt.hist(outlier_values, bins=20, alpha=0.8, color='red', 
                        label='Outliers', density=True)
                
                # Add vertical lines for mean and bounds
                mean_val = df[column].mean()
                std_val = df[column].std()
                lower_bound = mean_val - 3 * std_val
                upper_bound = mean_val + 3 * std_val
                
                plt.axvline(mean_val, color='green', linestyle='--', 
                           label=f'Mean: {mean_val:.3f}')
                plt.axvline(lower_bound, color='orange', linestyle=':', 
                           label=f'Lower bound (-3σ): {lower_bound:.3f}')
                plt.axvline(upper_bound, color='orange', linestyle=':', 
                           label=f'Upper bound (+3σ): {upper_bound:.3f}')
                
                plt.xlabel(column)
                plt.ylabel('Density')
                plt.title(f'Distribution of {column} in {file_name}\nOutliers highlighted in red')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Save the plot
                safe_filename = f"{file_name.replace('.csv', '')}_{column}_outliers.png"
                safe_filename = safe_filename.replace(' ', '_').replace('/', '_')
                plt.savefig(Path(output_dir) / safe_filename, dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"Error creating histogram for {file_name} - {column}: {e}")
    
    def create_cleaned_datasets(self, results, output_dir="Data"):
        """
        Create cleaned datasets by removing rows with missing articulation_rate values.
        
        Args:
            results (dict): Analysis results containing missing value information
            output_dir (str): Base directory to save cleaned datasets
        """
        print("\n🧹 Creating cleaned datasets...")
        
        # Define source and target folders
        source_folders = [
            "Acoustic Lines (with WER)",
            "Transformed Acoustic Lines (with WER)"
        ]
        
        target_folders = [
            "Acoustic Lines (with WER), Cleaned",
            "Transformed Acoustic Lines (with WER), Cleaned"
        ]
        
        # Process each folder pair
        for source_folder, target_folder in zip(source_folders, target_folders):
            source_path = Path(output_dir) / source_folder
            target_path = Path(output_dir) / target_folder
            
            if not source_path.exists():
                print(f"  Source folder not found: {source_folder}")
                continue
            
            # Create target directory
            target_path.mkdir(exist_ok=True)
            print(f"  Processing: {source_folder} → {target_folder}")
            
            # Get all CSV files in source folder
            csv_files = list(source_path.glob("*.csv"))
            
            total_rows_before = 0
            total_rows_after = 0
            total_missing_removed = 0
            
            for csv_file in csv_files:
                try:
                    # Read the data
                    df = pd.read_csv(csv_file)
                    rows_before = len(df)
                    total_rows_before += rows_before
                    
                    # Remove rows with missing articulation_rate
                    df_cleaned = df.dropna(subset=['articulation_rate'])
                    rows_after = len(df_cleaned)
                    total_rows_after += rows_after
                    
                    missing_removed = rows_before - rows_after
                    total_missing_removed += missing_removed
                    
                    # Save cleaned dataset
                    target_file = target_path / csv_file.name
                    df_cleaned.to_csv(target_file, index=False)
                    
                    if missing_removed > 0:
                        print(f"    {csv_file.name}: {rows_before} → {rows_after} rows (-{missing_removed} missing)")
                    else:
                        print(f"    {csv_file.name}: {rows_before} → {rows_after} rows (no changes)")
                        
                except Exception as e:
                    print(f"    Error processing {csv_file.name}: {e}")
            
            # Summary for this folder
            print(f"  {source_folder} Summary:")
            print(f"    Total rows before: {total_rows_before:,}")
            print(f"    Total rows after: {total_rows_after:,}")
            print(f"    Total missing rows removed: {total_missing_removed:,}")
            print(f"    Data retention: {(total_rows_after/total_rows_before)*100:.2f}%")
            print()
        
        print("✅ Cleaned datasets created successfully!")
        print("📁 New folders created:")
        for target_folder in target_folders:
            print(f"   • {target_folder}/")
    
    def run_comprehensive_analysis(self):
        """
        Run comprehensive analysis on all data folders.
        
        Returns:
            dict: Complete analysis results
        """
        print("Starting comprehensive data cleaning analysis...")
        print("=" * 60)
        
        all_files = self.get_all_csv_files()
        comprehensive_results = {
            'folder_summaries': {},
            'column_consistency': self.analyze_column_consistency(),
            'overall_statistics': {
                'total_files_analyzed': 0,
                'total_rows_analyzed': 0,
                'total_missing_values': 0,
                'total_outliers': 0
            },
            'missing_by_column': {},
            'outliers_by_column': {},
            'all_outliers': [],  # Store all outliers for ranking
            'missing_demographics': []  # Store demographic analysis of missing values
        }
        
        for folder, files in all_files.items():
            print(f"\nAnalyzing folder: {folder}")
            print("-" * 40)
            
            folder_results = {
                'files_analyzed': [],
                'total_rows': 0,
                'total_missing': 0,
                'total_outliers': 0
            }
            
            for file_path in files:
                try:
                    print(f"  Processing: {file_path.name}")
                    df = pd.read_csv(file_path)
                    
                    # Analyze this file
                    missing_analysis = self.analyze_missing_values(df, file_path.name)
                    missing_demographics = self.analyze_missing_values_demographics(df, file_path.name)
                    outlier_analysis = self.detect_outliers(df, file_path.name)
                    data_analysis = self.analyze_data_types_and_ranges(df, file_path.name)
                    
                    file_result = {
                        'file_name': file_path.name,
                        'rows': len(df),
                        'columns': len(df.columns),
                        'missing_values': missing_analysis,
                        'missing_demographics': missing_demographics,
                        'outliers': outlier_analysis,
                        'data_analysis': data_analysis
                    }
                    
                    folder_results['files_analyzed'].append(file_result)
                    folder_results['total_rows'] += len(df)
                    folder_results['total_missing'] += missing_analysis['total_missing']
                    folder_results['total_outliers'] += outlier_analysis['total_outliers']
                    
                    comprehensive_results['overall_statistics']['total_files_analyzed'] += 1
                    comprehensive_results['overall_statistics']['total_rows_analyzed'] += len(df)
                    comprehensive_results['overall_statistics']['total_missing_values'] += missing_analysis['total_missing']
                    comprehensive_results['overall_statistics']['total_outliers'] += outlier_analysis['total_outliers']
                    
                    # Aggregate missing values and outliers by column
                    for col, missing_count in missing_analysis['missing_by_column'].items():
                        if col not in comprehensive_results['missing_by_column']:
                            comprehensive_results['missing_by_column'][col] = 0
                        comprehensive_results['missing_by_column'][col] += missing_count
                    
                    for col, outlier_count in outlier_analysis['outliers_by_column'].items():
                        if col not in comprehensive_results['outliers_by_column']:
                            comprehensive_results['outliers_by_column'][col] = 0
                        comprehensive_results['outliers_by_column'][col] += outlier_count
                    
                    # Collect all outliers for ranking
                    comprehensive_results['all_outliers'].extend(outlier_analysis['top_outliers'])
                    
                    # Collect demographic analysis of missing values
                    comprehensive_results['missing_demographics'].append(missing_demographics)
                    
                except Exception as e:
                    print(f"    Error processing {file_path.name}: {e}")
                    file_result = {
                        'file_name': file_path.name,
                        'error': str(e)
                    }
                    folder_results['files_analyzed'].append(file_result)
            
            comprehensive_results['folder_summaries'][folder] = folder_results
            
        # Find top 5 outliers by z-score
        if comprehensive_results['all_outliers']:
            comprehensive_results['top_5_outliers'] = sorted(
                comprehensive_results['all_outliers'], 
                key=lambda x: x['z_score'], 
                reverse=True
            )[:5]
        
        return comprehensive_results
    
    def generate_cleaning_recommendations(self, results):
        """
        Generate cleaning recommendations based on analysis results.
        
        Args:
            results (dict): Analysis results
            
        Returns:
            dict: Cleaning recommendations
        """
        recommendations = {
            'missing_values': {},
            'outliers': {},
            'data_consistency': {}
        }
        
        # Missing value recommendations
        total_missing = results['overall_statistics']['total_missing_values']
        total_rows = results['overall_statistics']['total_rows_analyzed']
        
        if total_missing > 0:
            missing_percentage = (total_missing / total_rows) * 100
            recommendations['missing_values']['overall_percentage'] = missing_percentage
            
            if missing_percentage < 5:
                recommendations['missing_values']['strategy'] = 'Delete rows with missing values'
            elif missing_percentage < 20:
                recommendations['missing_values']['strategy'] = 'Impute missing values using median/mean'
            else:
                recommendations['missing_values']['strategy'] = 'Investigate missing value patterns'
        
        # Outlier recommendations
        total_outliers = results['overall_statistics']['total_outliers']
        if total_outliers > 0:
            outlier_percentage = (total_outliers / total_rows) * 100
            recommendations['outliers']['overall_percentage'] = outlier_percentage
            
            if outlier_percentage < 5:
                recommendations['outliers']['strategy'] = 'Remove outliers'
            elif outlier_percentage < 15:
                recommendations['outliers']['strategy'] = 'Winsorize or cap outliers'
            else:
                recommendations['outliers']['strategy'] = 'Investigate outlier patterns'
        
        # Column consistency recommendations
        column_consistency = results['column_consistency']
        if 'consistency_check' in column_consistency:
            unique_cols = column_consistency['consistency_check']['total_unique_columns']
            recommendations['data_consistency']['unique_columns'] = unique_cols
            
            if unique_cols <= 10:
                recommendations['data_consistency']['strategy'] = 'Columns are consistent across folders'
            else:
                recommendations['data_consistency']['strategy'] = 'Investigate column differences'
        
        return recommendations
    
    def print_summary_report(self, results, recommendations):
        """
        Print a comprehensive summary report.
        
        Args:
            results (dict): Analysis results
            recommendations (dict): Cleaning recommendations
        """
        print("\n" + "=" * 80)
        print("COMPREHENSIVE DATA CLEANING ANALYSIS REPORT")
        print("=" * 80)
        
        # Overall Statistics
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Total files analyzed: {results['overall_statistics']['total_files_analyzed']}")
        print(f"   Total rows analyzed: {results['overall_statistics']['total_rows_analyzed']:,}")
        print(f"   Total missing values: {results['overall_statistics']['total_missing_values']:,}")
        print(f"   Total outliers detected: {results['overall_statistics']['total_outliers']:,}")
        
        # Column Consistency
        print(f"\n🔍 COLUMN CONSISTENCY ANALYSIS:")
        column_check = results['column_consistency'].get('consistency_check', {})
        if 'total_unique_columns' in column_check:
            print(f"   Total unique columns across all folders: {column_check['total_unique_columns']}")
            print(f"   Unique columns: {', '.join(column_check.get('all_unique_columns', []))}")
        
        # Folder Summaries
        print(f"\n📁 FOLDER SUMMARIES:")
        for folder, summary in results['folder_summaries'].items():
            print(f"\n   {folder}:")
            print(f"     Files analyzed: {len(summary['files_analyzed'])}")
            print(f"     Total rows: {summary['total_rows']:,}")
            print(f"     Total missing: {summary['total_missing']:,}")
            print(f"     Total outliers: {summary['total_outliers']:,}")
        
        # Missing Values Analysis
        if results['overall_statistics']['total_missing_values'] > 0:
            print(f"\n❌ MISSING VALUES ANALYSIS:")
            missing_pct = (results['overall_statistics']['total_missing_values'] / 
                          results['overall_statistics']['total_rows_analyzed']) * 100
            print(f"   Overall missing percentage: {missing_pct:.2f}%")
            print(f"   Recommendation: {recommendations['missing_values'].get('strategy', 'N/A')}")
            
            # Show missing values by column
            print(f"   Missing values by column:")
            for col, missing_count in results['missing_by_column'].items():
                if missing_count > 0:
                    col_missing_pct = (missing_count / results['overall_statistics']['total_rows_analyzed']) * 100
                    print(f"     • {col}: {missing_count} ({col_missing_pct:.2f}%)")
            
            # Show missing values by demographics
            if 'missing_demographics' in results and results['missing_demographics']:
                print(f"   Missing values by demographics:")
                
                # Analyze by country
                country_missing = {}
                gender_missing = {}
                for demo in results['missing_demographics']:
                    if demo['missing_rows'] > 0:
                        country = demo['country']
                        gender = demo['gender']
                        
                        if country not in country_missing:
                            country_missing[country] = {'count': 0, 'total_rows': 0}
                        if gender not in gender_missing:
                            gender_missing[gender] = {'count': 0, 'total_rows': 0}
                        
                        country_missing[country]['count'] += demo['missing_rows']
                        country_missing[country]['total_rows'] += demo['total_rows']
                        gender_missing[gender]['count'] += demo['missing_rows']
                        gender_missing[gender]['total_rows'] += demo['total_rows']
                
                print(f"     By country:")
                for country, data in country_missing.items():
                    pct = (data['count'] / data['total_rows']) * 100
                    print(f"       • {country}: {data['count']} ({pct:.2f}%)")
                
                print(f"     By gender:")
                for gender, data in gender_missing.items():
                    pct = (data['count'] / data['total_rows']) * 100
                    print(f"       • {gender}: {data['count']} ({pct:.2f}%)")
        
        # Outliers Analysis
        if results['overall_statistics']['total_outliers'] > 0:
            print(f"\n🚨 OUTLIERS ANALYSIS:")
            outlier_pct = (results['overall_statistics']['total_outliers'] / 
                          results['overall_statistics']['total_rows_analyzed']) * 100
            print(f"   Overall outlier percentage: {outlier_pct:.2f}%")
            print(f"   Recommendation: {recommendations['outliers'].get('strategy', 'N/A')}")
            print(f"   Note: Outliers defined as > 3 standard deviations from mean (per file)")
            
            # Show outliers by column
            print(f"   Outliers by column:")
            for col, outlier_count in results['outliers_by_column'].items():
                if outlier_count > 0:
                    col_outlier_pct = (outlier_count / results['overall_statistics']['total_rows_analyzed']) * 100
                    print(f"     • {col}: {outlier_count} ({col_outlier_pct:.2f}%)")
        
        # Display top 5 outliers
        if 'top_5_outliers' in results and results['top_5_outliers']:
            print(f"\n🏆 TOP 5 BIGGEST OUTLIERS:")
            print(f"   Rank | Column | File | Value | Mean | Std | Z-Score | Deviation")
            print(f"   -----|--------|------|-------|------|-----|---------|-----------")
            for i, outlier in enumerate(results['top_5_outliers'], 1):
                print(f"   {i:2d}.  | {outlier['column']:6s} | {outlier['file'][:20]:20s} | {outlier['value']:6.3f} | {outlier['mean']:5.3f} | {outlier['std']:4.3f} | {outlier['z_score']:7.2f} | {outlier['deviation']:9.3f}")
        
        print("\n" + "=" * 80)
    
    def save_detailed_report(self, results, recommendations, output_file="data_cleaning_report.txt"):
        """
        Save detailed analysis report to a file.
        
        Args:
            results (dict): Analysis results
            recommendations (dict): Cleaning recommendations
            output_file (str): Output file path
        """
        with open(output_file, 'w') as f:
            f.write("DETAILED DATA CLEANING ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Write overall statistics
            f.write("OVERALL STATISTICS:\n")
            f.write("-" * 20 + "\n")
            for key, value in results['overall_statistics'].items():
                f.write(f"{key}: {value:,}\n")
            f.write("\n")
            
            # Write missing values by column
            if results['missing_by_column']:
                f.write("MISSING VALUES BY COLUMN:\n")
                f.write("-" * 25 + "\n")
                for col, missing_count in results['missing_by_column'].items():
                    if missing_count > 0:
                        col_missing_pct = (missing_count / results['overall_statistics']['total_rows_analyzed']) * 100
                        f.write(f"  {col}: {missing_count} ({col_missing_pct:.2f}%)\n")
                f.write("\n")
            
            # Write missing values by demographics
            if 'missing_demographics' in results and results['missing_demographics']:
                f.write("MISSING VALUES BY DEMOGRAPHICS:\n")
                f.write("-" * 30 + "\n")
                
                # Analyze by country
                country_missing = {}
                gender_missing = {}
                for demo in results['missing_demographics']:
                    if demo['missing_rows'] > 0:
                        country = demo['country']
                        gender = demo['gender']
                        
                        if country not in country_missing:
                            country_missing[country] = {'count': 0, 'total_rows': 0}
                        if gender not in gender_missing:
                            gender_missing[gender] = {'count': 0, 'total_rows': 0}
                        
                        country_missing[country]['count'] += demo['missing_rows']
                        country_missing[country]['total_rows'] += demo['total_rows']
                        gender_missing[gender]['count'] += demo['missing_rows']
                        gender_missing[gender]['total_rows'] += demo['total_rows']
                
                f.write("By country:\n")
                for country, data in country_missing.items():
                    pct = (data['count'] / data['total_rows']) * 100
                    f.write(f"  {country}: {data['count']} ({pct:.2f}%)\n")
                
                f.write("\nBy gender:\n")
                for gender, data in gender_missing.items():
                    pct = (data['count'] / data['total_rows']) * 100
                    f.write(f"  {gender}: {data['count']} ({pct:.2f}%)\n")
                f.write("\n")
            
            # Write outliers by column
            if results['outliers_by_column']:
                f.write("OUTLIERS BY COLUMN:\n")
                f.write("-" * 20 + "\n")
                for col, outlier_count in results['outliers_by_column'].items():
                    if outlier_count > 0:
                        col_outlier_pct = (outlier_count / results['overall_statistics']['total_rows_analyzed']) * 100
                        f.write(f"  {col}: {outlier_count} ({col_outlier_pct:.2f}%)\n")
                f.write("\n")
            
            # Write top 5 outliers
            if 'top_5_outliers' in results and results['top_5_outliers']:
                f.write("TOP 5 BIGGEST OUTLIERS:\n")
                f.write("-" * 25 + "\n")
                f.write("Rank | Column | File | Value | Mean | Std | Z-Score | Deviation\n")
                f.write("-----|--------|------|-------|------|-----|---------|-----------\n")
                for i, outlier in enumerate(results['top_5_outliers'], 1):
                    f.write(f"{i:2d}.  | {outlier['column']:6s} | {outlier['file'][:20]:20s} | {outlier['value']:6.3f} | {outlier['mean']:5.3f} | {outlier['std']:4.3f} | {outlier['z_score']:7.2f} | {outlier['deviation']:9.3f}\n")
                f.write("\n")
            
            # Write folder summaries
            f.write("FOLDER SUMMARIES:\n")
            f.write("-" * 20 + "\n")
            for folder, summary in results['folder_summaries'].items():
                f.write(f"\n{folder}:\n")
                for key, value in summary.items():
                    if key != 'files_analyzed':
                        f.write(f"  {key}: {value:,}\n")
            
            # Write recommendations
            f.write("\nCLEANING RECOMMENDATIONS:\n")
            f.write("-" * 25 + "\n")
            for category, recs in recommendations.items():
                f.write(f"\n{category.upper()}:\n")
                if isinstance(recs, dict):
                    for key, value in recs.items():
                        f.write(f"  {key}: {value}\n")
                elif isinstance(recs, list):
                    for item in recs:
                        f.write(f"  • {item}\n")
        
        print(f"\n📄 Detailed report saved to: {output_file}")


def main():
    """Main function to run the data cleaning analysis."""
    print("🚀 Starting ASR Data Cleaning Analysis...")
    
    # Initialize analyzer
    analyzer = DataCleaningAnalyzer()
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis()
    
    # Generate recommendations
    recommendations = analyzer.generate_cleaning_recommendations(results)
    
    # Print summary report
    analyzer.print_summary_report(results, recommendations)
    
    # Save detailed report
    analyzer.save_detailed_report(results, recommendations)
    
    # Create outlier histograms
    print("\n📊 Creating outlier histograms...")
    analyzer.create_outlier_histograms(results)
    
    # Create cleaned datasets
    print("\n🧹 Creating cleaned datasets...")
    analyzer.create_cleaned_datasets(results)
    
    print("\n✅ Analysis complete! Check the generated report for detailed information.")


if __name__ == "__main__":
    main() 