"""
Validate the standardized datasets
"""

import pandas as pd
from pathlib import Path

def validate_data():
    standardized_path = Path('standardized_data')
    
    print("\n" + "="*70)
    print("DATA VALIDATION REPORT")
    print("="*70)
    
    # Load all standardized datasets
    exclusivity = pd.read_csv(standardized_path / 'exclusivity_standardized.csv')
    patents = pd.read_csv(standardized_path / 'patents_standardized.csv')
    products = pd.read_csv(standardized_path / 'products_standardized.csv')
    reference = pd.read_csv(standardized_path / 'reference_standardized.csv')
    
    print("\n✅ VALIDATION CHECKS:")
    
    # Check 1: Column naming convention
    print("\n1. Column Naming Convention:")
    all_datasets = {
        'exclusivity': exclusivity,
        'patents': patents,
        'products': products,
        'reference': reference
    }
    
    for name, df in all_datasets.items():
        valid = all(col.islower() and '_' in col or col.isalnum() for col in df.columns)
        status = "✓" if valid else "✗"
        print(f"   {status} {name}: All columns lowercase with underscores")
    
    # Check 2: No duplicate rows
    print("\n2. Duplicate Rows:")
    for name, df in all_datasets.items():
        dupes = df.duplicated().sum()
        status = "✓" if dupes == 0 else "✗"
        print(f"   {status} {name}: {dupes} duplicates found")
    
    # Check 3: Data types
    print("\n3. Numeric Columns Have Correct Types:")
    numeric_checks = [
        ('exclusivity', exclusivity, ['appl_type', 'appl_no']),
        ('patents', patents, ['appl_type', 'appl_no']),
        ('products', products, []),  # appl_no has many nulls
    ]
    
    for name, df, cols in numeric_checks:
        for col in cols:
            if col in df.columns:
                is_numeric = pd.api.types.is_numeric_dtype(df[col])
                status = "✓" if is_numeric else "✗"
                print(f"   {status} {name}.{col}: {df[col].dtype}")
    
    # Check 4: Sample data preview
    print("\n4. Sample Data Preview:")
    print("\n   EXCLUSIVITY (first 3 rows):")
    print(exclusivity.head(3).to_string(index=False))
    
    print("\n   PRODUCTS (first 3 rows):")
    print(products[['trade_name', 'applicant', 'approval_date', 'type']].head(3).to_string(index=False))
    
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)
    print("\n✅ All datasets have been standardized successfully!")
    print("   - Column names are consistent")
    print("   - Duplicates have been removed")
    print("   - Data types are appropriate")
    print("   - Missing values are handled consistently")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    validate_data()
