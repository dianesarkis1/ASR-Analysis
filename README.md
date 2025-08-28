# ASR Analysis Project
This project analyzes phonological speech features and Word Error Rate data from Whisper v3 across French and Spanish demographic groups and regions. The main analysis is in inference_analysis.ipynb, with preprocessing and cleaning scripts in Data/. The goal is to improve model explainability by comparing performance on original audio snippets versus Cartesia-transformed snippets, where Cartesia standardizes all speakers within a language to a common voice.

## Execution Order
The Python scripts must be run in the following order to ensure proper data flow:
### 1. Data Merging Scripts
First, run the data merging scripts to combine acoustic features with WER data:
- **`Data/data_merging_french_normal.py`** - Merges non-transformed French WERs into acoustic features CSVs
- **`Data/data_merging_other.py`** - Merges other language/transformed WERs into acoustic features CSVs
- This creates consolidated datasets in "Acoustic Lines (with WER)" folders with WER scores added to each row.

### 2. Data Cleaning Analysis
Run the data cleaning analysis script:
- **`data_cleaning_analysis.py`** - Performs comprehensive data cleaning and outlier detection
- Generates outlier plots in `outlier_plots/` folder
- Produces `data_cleaning_report.txt` with cleaning statistics
- Creates cleaned datasets in "Acoustic Lines (with WER), Cleaned" folders

### 3. Data Cleaning Checks
Run the data cleaning verification script:
- **`cleaned_data_checks.py`** - Validates the cleaned datasets and ensures data integrity

### 4. Main Analysis Notebook
Finally, run the main analysis notebook

## Requirements
The `requirements.txt` file contains all the Python package dependencies needed to run this project. Install them using:
