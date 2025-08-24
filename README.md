# ASR Analysis Project

This repository contains scripts for analyzing Automatic Speech Recognition (ASR) data, including acoustic features and Word Error Rate (WER) scores across different countries, genders, and languages.

## 📁 Project Structure

```
ASR-Analysis/
├── Data/
│   ├── Acoustic Lines/                    # Original acoustic features
│   ├── Acoustic Lines (with WER)/         # Acoustic features + WER scores
│   ├── Transformed Acoustic Lines/        # Transformed acoustic features
│   ├── Transformed Acoustic Lines (with WER)/ # Transformed features + WER
│   ├── WERs/                             # Word Error Rate files
│   ├── data_merging_french_normal.py      # French data merging script
│   └── data_merging_other.py             # Other languages data merging script
├── outlier_plots/                         # Generated outlier analysis plots
├── data_cleaning_analysis.py              # Comprehensive data cleaning script
├── data_cleaning_report.txt               # Generated cleaning analysis report
├── requirements.txt                       # Python dependencies
└── README.md                              # This file
```

## 🚀 Script Execution Order

**IMPORTANT**: The scripts must be run in the following order to ensure proper data processing:

### 1. Data Merging Scripts (First)
Run these scripts to prepare and merge your raw data:

#### **French Data Merging**
```bash
cd Data
python data_merging_french_normal.py
```

#### **Other Languages Data Merging**
```bash
python data_merging_other.py
```

**What these scripts do:**
- Process raw acoustic feature data
- Merge data from different sources
- Create the four main data folders:
  - `Acoustic Lines/`
  - `Acoustic Lines (with WER)/`
  - `Transformed Acoustic Lines/`
  - `Transformed Acoustic Lines (with WER)/`

### 2. Data Cleaning Analysis Script (Second)
After the data merging is complete, run the comprehensive cleaning analysis:

```bash
cd ..  # Return to project root
python data_cleaning_analysis.py
```

**What this script does:**
- Analyzes all four data folders for missing values and outliers
- Provides demographic breakdown of missing values (by country, gender)
- Detects outliers using 3σ method (per file per variable)
- Generates outlier histograms for visual analysis
- Creates comprehensive cleaning recommendations
- Saves detailed report to `data_cleaning_report.txt`
- Generates outlier plots in `outlier_plots/` directory

## 📊 What You'll Get

### **Data Cleaning Report**
- Overall statistics (files, rows, missing values, outliers)
- Missing values by column and demographics
- Outliers by column with percentages
- Top 5 biggest outliers with details
- Folder summaries
- Cleaning recommendations

### **Outlier Analysis**
- 80+ histogram plots showing:
  - Distribution of all values for each variable
  - Outliers highlighted in red
  - Mean and ±3σ bounds marked
  - One plot per file per variable that has outliers

### **Key Insights**
- **Missing Values**: All 30 missing values are in `articulation_rate` column
- **Demographic Bias**: Canada (2.31%) and male speakers (1.59%) have higher missing rates
- **Outliers**: 6.18% of data contains outliers, with `pitch_std_dev` having the most (2.07%)

## 🛠️ Prerequisites

Install the required Python packages:

```bash
pip install -r requirements.txt
```

**Required packages:**
- pandas >= 1.3.0
- numpy == 1.22.3
- matplotlib == 3.5.3
- seaborn == 0.11.2
- pathlib2 >= 2.3.0

## ⚠️ Important Notes

1. **Data Merging First**: Always run the data merging scripts before the cleaning analysis
2. **Bias Awareness**: Missing values are not randomly distributed - removing them will bias your sample
3. **Outlier Investigation**: 6.18% outlier rate suggests investigating patterns rather than simple removal
4. **File Dependencies**: The cleaning script expects the four data folders to exist and contain CSV files

## 🔍 Understanding the Output

### **Missing Values Analysis**
- **Low overall rate**: 0.24% missing data
- **Concentrated issue**: All missing values in `articulation_rate`
- **Demographic patterns**: Higher rates in Canada and male speakers

### **Outlier Analysis**
- **Definition**: Values > 3 standard deviations from mean (per file)
- **Manageable rate**: 6.18% of data
- **Most problematic**: `pitch_std_dev` column

### **Recommendations**
- **Missing values**: Consider imputation rather than deletion
- **Outliers**: Investigate patterns, consider winsorizing
- **Data consistency**: All folders have the same 10 columns

## 📈 Next Steps

After running the cleaning analysis:

1. **Review the report** (`data_cleaning_report.txt`)
2. **Examine outlier plots** in `outlier_plots/` directory
3. **Make informed decisions** about handling missing values and outliers
4. **Consider bias implications** before removing data
5. **Plan data cleaning strategy** based on the analysis results

## 🤝 Contributing

When adding new scripts or modifying existing ones:
1. Follow the established naming convention
2. Update this README if the workflow changes
3. Test the execution order
4. Commit and push changes

## 📞 Support

If you encounter issues:
1. Check that scripts are run in the correct order
2. Verify all required packages are installed
3. Ensure the Data/ folder structure is correct
4. Check the generated reports for error messages 