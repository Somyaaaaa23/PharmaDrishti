# Dataset Standardization Documentation

## Overview
This document describes the standardization process applied to the PharmaDrishti datasets located in the `archive/` folder.

## Standardized Datasets

### Location
- **Original Data**: `archive/`
- **Standardized Data**: `standardized_data/`

### Files Processed
1. `exclusivity.csv` → `exclusivity_standardized.csv`
2. `patents.csv` → `patents_standardized.csv`
3. `products.csv` → `products_standardized.csv`
4. `reference.csv` → `reference_standardized.csv`

## Standardization Steps Applied

### 1. Column Name Standardization
- Converted all column names to lowercase
- Replaced spaces with underscores
- Removed special characters
- Example: `Appl_No` → `appl_no`, `DF;Route` → `dfroute`

### 2. Whitespace Cleaning
- Removed leading and trailing whitespace from all text fields
- Collapsed multiple spaces into single spaces
- Ensures consistent text formatting

### 3. Date Standardization
- Standardized date formats to `YYYY-MM-DD`
- Handled various input formats:
  - `Oct 7, 2014` → `2014-10-07`
  - `Approved Prior to Jan 1, 1982` → `1982-01-01`
- Columns affected: `exclusivity_date`, `patent_expire_date_text`, `approval_date`

### 4. Missing Value Handling
- Standardized all missing value representations to `NaN`
- Replaced empty strings, 'NA', 'N/A', 'null', 'NULL', 'None' with `NaN`
- Provides consistent null handling across all datasets

### 5. Data Type Conversion
- Converted numeric columns to appropriate types (int64, float64)
- Numeric columns: `appl_no`, `product_no`, `patent_no`, `appl_type`
- Maintains data integrity and enables proper numerical operations

### 6. Duplicate Removal
- Removed duplicate rows from datasets
- **Exclusivity**: 117 duplicates removed (1,672 → 1,555 rows)
- **Patents**: 797 duplicates removed (13,149 → 12,352 rows)
- **Products**: 0 duplicates (33,180 rows maintained)
- **Reference**: 0 duplicates (922 rows maintained)

## Dataset Statistics

### Exclusivity Dataset
- **Rows**: 1,555 (after removing 117 duplicates)
- **Columns**: 5
- **Key Fields**: `appl_type`, `appl_no`, `product_no`, `exclusivity_code`, `exclusivity_date`

### Patents Dataset
- **Rows**: 12,352 (after removing 797 duplicates)
- **Columns**: 9
- **Key Fields**: `appl_type`, `appl_no`, `product_no`, `patent_no`, `patent_expire_date_text`

### Products Dataset
- **Rows**: 33,180
- **Columns**: 14
- **Key Fields**: `ingredient`, `trade_name`, `applicant`, `strength`, `approval_date`, `type`

### Reference Dataset
- **Rows**: 922
- **Columns**: 2
- **Key Fields**: `exclusivity_code`, `code_meaning`

## Usage

### Running Standardization
```bash
cd PharmaDrishti
python3 standardize_dataset.py
```

### Generating Report
```bash
python3 generate_standardization_report.py
```

### Validating Data
```bash
python3 validate_standardized_data.py
```

## Data Quality Improvements

✅ **Consistency**: All column names follow the same naming convention
✅ **Cleanliness**: Removed whitespace and formatting issues
✅ **Completeness**: Standardized missing value representation
✅ **Accuracy**: Removed duplicate records
✅ **Usability**: Proper data types for analysis and modeling

## Integration with Existing Code

The standardized datasets can be used with the existing PharmaDrishti application by updating the data loader to point to the `standardized_data/` folder instead of `archive/`.

### Recommended Changes to `data_loader.py`:
```python
# Change from:
df = pd.read_csv('archive/products.csv')

# To:
df = pd.read_csv('standardized_data/products_standardized.csv')
```

## Notes

- Original data in `archive/` folder remains unchanged
- Standardized data is saved in a separate `standardized_data/` folder
- All transformations are documented and reproducible
- The standardization process can be re-run at any time

## Contact

For questions or issues with the standardization process, please refer to the main project documentation.
