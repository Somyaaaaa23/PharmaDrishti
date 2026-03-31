"""
Generate a detailed standardization report
"""

import pandas as pd
from pathlib import Path

def generate_report():
    archive_path = Path('archive')
    standardized_path = Path('standardized_data')
    
    print("\n" + "="*70)
    print("DATASET STANDARDIZATION REPORT")
    print("="*70)
    
    datasets = ['exclusivity', 'patents', 'products', 'reference']
    
    for dataset in datasets:
        print(f"\n{'='*70}")
        print(f"{dataset.upper()}.CSV")
        print(f"{'='*70}")
        
        # Load original and standardized
        original = pd.read_csv(archive_path / f'{dataset}.csv')
        standardized = pd.read_csv(standardized_path / f'{dataset}_standardized.csv')
        
        print(f"\n📊 SIZE COMPARISON:")
        print(f"  Original rows:      {len(original):,}")
        print(f"  Standardized rows:  {len(standardized):,}")
        print(f"  Rows removed:       {len(original) - len(standardized):,}")
        
        print(f"\n📋 COLUMN CHANGES:")
        print(f"  Original columns:   {list(original.columns)}")
        print(f"  Standardized cols:  {list(standardized.columns)}")
        
        print(f"\n🔍 DATA QUALITY:")
        print(f"  Missing values before: {original.isna().sum().sum():,}")
        print(f"  Missing values after:  {standardized.isna().sum().sum():,}")
        
        print(f"\n📈 COLUMN DETAILS:")
        for col in standardized.columns:
            null_count = standardized[col].isna().sum()
            null_pct = (null_count / len(standardized)) * 100
            dtype = standardized[col].dtype
            print(f"  {col:30s} | Type: {str(dtype):10s} | Nulls: {null_count:5d} ({null_pct:5.1f}%)")
    
    print("\n" + "="*70)
    print("STANDARDIZATION IMPROVEMENTS")
    print("="*70)
    print("""
✅ Column names: Lowercase, underscores, no special characters
✅ Whitespace: Removed leading/trailing spaces
✅ Dates: Standardized to YYYY-MM-DD format
✅ Missing values: Consistent NaN representation
✅ Duplicates: Removed duplicate rows
✅ Numeric fields: Proper data types
✅ Text fields: Cleaned extra spaces
    """)
    
    print("="*70)
    print("FILES LOCATION")
    print("="*70)
    print(f"Original files:     {archive_path}/")
    print(f"Standardized files: {standardized_path}/")
    print("="*70 + "\n")

if __name__ == "__main__":
    generate_report()
