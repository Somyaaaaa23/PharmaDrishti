"""
Dataset Standardization Script for PharmaDrishti
Standardizes all CSV files in the archive folder
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from datetime import datetime

class DatasetStandardizer:
    def __init__(self, archive_path='archive'):
        self.archive_path = Path(archive_path)
        self.output_path = Path('standardized_data')
        self.output_path.mkdir(exist_ok=True)
    
    def standardize_column_names(self, df):
        """Standardize column names: lowercase, underscores, no special chars"""
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.lower()
        df.columns = df.columns.str.replace(r'[^\w\s]', '', regex=True)
        df.columns = df.columns.str.replace(r'\s+', '_', regex=True)
        return df
    
    def clean_whitespace(self, df):
        """Remove leading/trailing whitespace from string columns"""
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip() if df[col].dtype == 'object' else df[col]
        return df

    def standardize_dates(self, df, date_columns):
        """Standardize date formats to YYYY-MM-DD"""
        for col in date_columns:
            if col in df.columns:
                df[col] = df[col].apply(self._parse_date)
        return df
    
    def _parse_date(self, date_str):
        """Parse various date formats"""
        if pd.isna(date_str) or date_str == '':
            return None
        
        date_str = str(date_str).strip()
        
        # Handle "Approved Prior to Jan 1, 1982"
        if 'Approved Prior' in date_str or 'prior' in date_str.lower():
            return '1982-01-01'
        
        # Try common formats
        formats = [
            '%b %d, %Y',  # Oct 7, 2014
            '%B %d, %Y',  # October 7, 2014
            '%m/%d/%Y',   # 10/07/2014
            '%Y-%m-%d',   # 2014-10-07
            '%d-%b-%Y',   # 07-Oct-2014
        ]
        
        for fmt in formats:
            try:
                return pd.to_datetime(date_str, format=fmt).strftime('%Y-%m-%d')
            except:
                continue
        
        return date_str  # Return original if can't parse

    def handle_missing_values(self, df):
        """Standardize missing values representation"""
        # Replace empty strings and various null representations with NaN
        df = df.replace(['', ' ', 'NA', 'N/A', 'null', 'NULL', 'None'], np.nan)
        return df
    
    def standardize_numeric_columns(self, df, numeric_columns):
        """Convert specified columns to numeric types"""
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    
    def remove_duplicates(self, df):
        """Remove duplicate rows"""
        initial_count = len(df)
        df = df.drop_duplicates()
        removed = initial_count - len(df)
        if removed > 0:
            print(f"  Removed {removed} duplicate rows")
        return df
    
    def standardize_text_fields(self, df):
        """Standardize text fields: consistent casing, remove extra spaces"""
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].dtype == 'object':
                # Remove multiple spaces
                df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
        return df

    def standardize_exclusivity(self):
        """Standardize exclusivity.csv"""
        print("\nStandardizing exclusivity.csv...")
        df = pd.read_csv(self.archive_path / 'exclusivity.csv')
        
        # Drop index column if exists
        if 'index' in df.columns:
            df = df.drop('index', axis=1)
        
        # Standardize column names
        df = self.standardize_column_names(df)
        
        # Clean whitespace
        df = self.clean_whitespace(df)
        
        # Standardize dates
        df = self.standardize_dates(df, ['exclusivity_date'])
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Standardize numeric columns
        numeric_cols = ['appl_no', 'product_no']
        df = self.standardize_numeric_columns(df, numeric_cols)
        
        # Remove duplicates
        df = self.remove_duplicates(df)
        
        # Save
        output_file = self.output_path / 'exclusivity_standardized.csv'
        df.to_csv(output_file, index=False)
        print(f"  Saved to {output_file}")
        print(f"  Shape: {df.shape}")
        return df

    def standardize_patents(self):
        """Standardize patents.csv"""
        print("\nStandardizing patents.csv...")
        df = pd.read_csv(self.archive_path / 'patents.csv')
        
        # Drop index column if exists
        if 'index' in df.columns:
            df = df.drop('index', axis=1)
        
        # Standardize column names
        df = self.standardize_column_names(df)
        
        # Clean whitespace
        df = self.clean_whitespace(df)
        
        # Standardize dates
        df = self.standardize_dates(df, ['patent_expire_date_text'])
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Standardize numeric columns
        numeric_cols = ['appl_no', 'product_no', 'patent_no']
        df = self.standardize_numeric_columns(df, numeric_cols)
        
        # Remove duplicates
        df = self.remove_duplicates(df)
        
        # Standardize text fields
        df = self.standardize_text_fields(df)
        
        # Save
        output_file = self.output_path / 'patents_standardized.csv'
        df.to_csv(output_file, index=False)
        print(f"  Saved to {output_file}")
        print(f"  Shape: {df.shape}")
        return df

    def standardize_products(self):
        """Standardize products.csv"""
        print("\nStandardizing products.csv...")
        df = pd.read_csv(self.archive_path / 'products.csv')
        
        # Drop index column if exists
        if 'index' in df.columns:
            df = df.drop('index', axis=1)
        
        # Standardize column names
        df = self.standardize_column_names(df)
        
        # Clean whitespace
        df = self.clean_whitespace(df)
        
        # Standardize dates
        df = self.standardize_dates(df, ['approval_date'])
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Standardize numeric columns
        numeric_cols = ['appl_no', 'product_no']
        df = self.standardize_numeric_columns(df, numeric_cols)
        
        # Remove duplicates
        df = self.remove_duplicates(df)
        
        # Standardize text fields
        df = self.standardize_text_fields(df)
        
        # Save
        output_file = self.output_path / 'products_standardized.csv'
        df.to_csv(output_file, index=False)
        print(f"  Saved to {output_file}")
        print(f"  Shape: {df.shape}")
        return df

    def standardize_reference(self):
        """Standardize reference.csv"""
        print("\nStandardizing reference.csv...")
        df = pd.read_csv(self.archive_path / 'reference.csv')
        
        # Drop index column if exists
        if 'index' in df.columns:
            df = df.drop('index', axis=1)
        
        # Standardize column names
        df = self.standardize_column_names(df)
        
        # Clean whitespace
        df = self.clean_whitespace(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Remove duplicates
        df = self.remove_duplicates(df)
        
        # Standardize text fields
        df = self.standardize_text_fields(df)
        
        # Save
        output_file = self.output_path / 'reference_standardized.csv'
        df.to_csv(output_file, index=False)
        print(f"  Saved to {output_file}")
        print(f"  Shape: {df.shape}")
        return df
    
    def run_all(self):
        """Run standardization on all datasets"""
        print("="*60)
        print("Starting Dataset Standardization")
        print("="*60)
        
        results = {}
        results['exclusivity'] = self.standardize_exclusivity()
        results['patents'] = self.standardize_patents()
        results['products'] = self.standardize_products()
        results['reference'] = self.standardize_reference()
        
        print("\n" + "="*60)
        print("Standardization Complete!")
        print("="*60)
        print(f"\nStandardized files saved to: {self.output_path}/")
        
        return results

if __name__ == "__main__":
    standardizer = DatasetStandardizer()
    results = standardizer.run_all()
