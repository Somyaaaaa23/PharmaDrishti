# PharmaDrishti - Complete File Importance Guide

## 📚 Overview
This document explains the purpose and importance of every file in the PharmaDrishti project.

---

## 🎯 CORE APPLICATION FILES

### 1. **dashboard.py** ⭐⭐⭐⭐⭐
**Purpose**: Main Streamlit web application - the heart of PharmaDrishti

**What it does**:
- Creates the interactive web interface users see
- Handles all user inputs (disease, price, sliders)
- Calculates suitability scores for all personas
- Generates visualizations (gauge charts, bar charts)
- Integrates with Google Gemini AI for insights
- Compares with real market data
- Exports results to CSV

**Why it's important**: This is the entry point - without this, users can't interact with the system. It orchestrates all other modules.

**Key features**:
- 911 lines of code
- Custom CSS styling for professional look
- Real-time predictions (<5 seconds)
- AI-powered recommendations
- Market comparison analysis

---

### 2. **data_loader.py** ⭐⭐⭐⭐
**Purpose**: Loads persona data from JSON files

**What it does**:
- Reads `indian_healthcare_personas.json`
- Converts JSON to pandas DataFrame
- Handles errors gracefully (file not found, invalid JSON)
- Returns empty DataFrame if loading fails

**Why it's important**: All predictions depend on persona data. Without this, the system has no one to analyze.

**Key features**:
- Simple, focused module (60 lines)
- Robust error handling
- Path handling for different environments

---

### 3. **suitability_calculator.py** ⭐⭐⭐⭐⭐
**Purpose**: Core algorithm that calculates adoption probability

**What it does**:
- Implements the weighted scoring formula:
  - Disease Match (30%)
  - Affordability (25%)
  - Market Pricing (20%)
  - Manufacturing (15%)
  - Risk Tolerance (10%)
- Applies sigmoid transformation for probability
- Creates interaction datasets for ML training
- Includes market pricing competitiveness analysis

**Why it's important**: This is the "brain" - the algorithm that determines if a medicine will succeed. All predictions come from here.

**Key features**:
- 250+ lines of sophisticated logic
- Considers 5 different factors
- Market-aware pricing analysis
- Used for both real-time predictions AND training data generation

---

### 4. **medicine_generator.py** ⭐⭐⭐
**Purpose**: Generates synthetic medicine profiles for training

**What it does**:
- Creates 5,000 realistic medicine profiles
- Uses statistical distributions (log-normal for price, beta for risk)
- Covers 10 disease categories
- Generates unique medicine IDs

**Why it's important**: Needed to create the 50M training dataset. Without diverse medicines, the ML model can't learn patterns.

**Key features**:
- Realistic price distribution (₹5K-₹500K)
- Risk skewed toward safer medicines (realistic)
- Reproducible (seed=42)

---

## 🤖 MACHINE LEARNING FILES

### 5. **train_model.py** ⭐⭐⭐⭐⭐
**Purpose**: Trains the XGBoost prediction model

**What it does**:
- Loads 10,000 personas
- Generates 5,000 medicines
- Creates 50,000,000 persona-medicine interactions
- Trains XGBoost regressor
- Saves trained model to `adoption_model.pkl`
- Saves feature names to `feature_names.pkl`

**Why it's important**: Creates the AI model that powers predictions. Without this, you'd need to calculate scores manually for every query.

**Training specs**:
- 15-30 minutes on standard laptop
- Requires 8-16GB RAM
- Achieves R² > 0.85
- Model size: ~50-100 MB

---

### 6. **adoption_model.pkl** ⭐⭐⭐⭐⭐
**Purpose**: Trained XGBoost model (binary file)

**What it contains**:
- Trained decision trees
- Feature importance weights
- Model hyperparameters

**Why it's important**: This IS the AI. Dashboard loads this to make instant predictions. Without it, you must retrain (30 minutes).

**Usage**: Loaded once at startup, cached in memory for fast predictions.

---

### 7. **feature_names.pkl** ⭐⭐⭐
**Purpose**: Stores feature column names

**What it contains**:
- List of feature names in correct order
- Ensures consistency between training and prediction

**Why it's important**: ML models are sensitive to feature order. This ensures predictions use the same features as training.

---

## 📊 DATA FILES

### 8. **indian_healthcare_personas.json** ⭐⭐⭐⭐⭐
**Purpose**: 10,000 synthetic Indian healthcare personas

**What it contains**:
- Demographics: age, gender, location
- Financial: income, healthcare spend
- Clinical: diseases, conditions
- Behavioral: price sensitivity, preferences

**Why it's important**: The "market" the system analyzes. Every prediction is based on these personas.

**Size**: ~2-3 MB JSON file

---

### 9. **archive/** folder ⭐⭐⭐
**Purpose**: Original FDA datasets (exclusivity, patents, products, reference)

**What it contains**:
- `exclusivity.csv`: Drug exclusivity data (1,672 rows)
- `patents.csv`: Patent information (13,149 rows)
- `products.csv`: Product details (33,180 rows)
- `reference.csv`: Reference codes (922 rows)

**Why it's important**: Real pharmaceutical data for validation and future features (competitive analysis, patent expiry tracking).

**Current status**: Not actively used in MVP, but valuable for Phase 2 features.

---

### 10. **standardized_data/** folder ⭐⭐⭐⭐
**Purpose**: Cleaned, standardized versions of archive data

**What it contains**:
- `exclusivity_standardized.csv` (1,555 rows, 117 duplicates removed)
- `patents_standardized.csv` (12,352 rows, 797 duplicates removed)
- `products_standardized.csv` (33,180 rows)
- `reference_standardized.csv` (922 rows)

**Why it's important**: Ready-to-use data with:
- Consistent column names (lowercase, underscores)
- Standardized dates (YYYY-MM-DD)
- No duplicates
- Proper data types

**Created by**: `standardize_dataset.py`

---

## 📖 DOCUMENTATION FILES

### 11. **README.md** ⭐⭐⭐⭐⭐
**Purpose**: Project overview and quick start guide

**What it contains**:
- Executive summary
- Quick start instructions
- Feature overview
- System metrics
- Roadmap
- Troubleshooting

**Why it's important**: First file anyone reads. Explains what PharmaDrishti is and how to use it.

**Audience**: Everyone (developers, users, stakeholders)

---

### 12. **DASHBOARD_GUIDE.md** ⭐⭐⭐⭐
**Purpose**: Complete dashboard user manual

**What it contains**:
- How to use every feature
- Input parameter explanations
- Score interpretation
- Tips for best results
- Troubleshooting

**Why it's important**: Helps users understand what each button/slider does and how to interpret results.

**Audience**: End users (product managers, analysts)

---

### 13. **DASHBOARD_SETUP.md** ⭐⭐⭐⭐
**Purpose**: Technical setup instructions

**What it contains**:
- Installation steps
- Dependency installation
- Environment configuration
- Gemini API setup
- Running the dashboard

**Why it's important**: Gets new users from zero to running dashboard in 5 minutes.

**Audience**: Technical users, developers

---

### 14. **FEATURES.md** ⭐⭐⭐⭐
**Purpose**: Detailed feature documentation

**What it contains**:
- Every feature explained in depth
- How scoring algorithm works
- Component breakdown
- AI insights explanation
- Export functionality

**Why it's important**: Deep dive into how everything works. Reference for understanding the system.

**Audience**: Power users, developers, stakeholders

---

### 15. **METRICS_AND_FORMULAS.md** ⭐⭐⭐⭐
**Purpose**: Mathematical formulas and calculations

**What it contains**:
- Suitability score formula
- Component weight justifications
- Statistical methods
- Performance metrics

**Why it's important**: Transparency - shows exactly how predictions are calculated. Critical for trust and validation.

**Audience**: Data scientists, technical stakeholders

---

### 16. **TECHNICAL_ANALYSIS.md** ⭐⭐⭐⭐
**Purpose**: Technical architecture documentation

**What it contains**:
- System architecture
- Technology stack
- Data flow diagrams
- Performance optimization
- Security measures

**Why it's important**: Blueprint for developers. Explains how the system is built.

**Audience**: Developers, architects

---

### 17. **QUICK_REFERENCE.md** ⭐⭐⭐
**Purpose**: One-page cheat sheet

**What it contains**:
- Quick commands
- Key metrics
- Common issues
- Fast troubleshooting

**Why it's important**: Quick answers without reading full docs.

**Audience**: All users

---

### 18. **requirements.md** ⭐⭐⭐⭐
**Purpose**: Product Requirements Document (PRD)

**What it contains**:
- Functional requirements
- Non-functional requirements
- Success criteria
- Constraints and assumptions

**Why it's important**: Defines WHAT the system should do (not HOW). Used for planning and validation.

**Audience**: Product managers, stakeholders

---

### 19. **design.md** ⭐⭐⭐⭐
**Purpose**: Technical Design Document (TDD)

**What it contains**:
- System architecture
- Data models
- API specifications
- ML pipeline design
- Security design

**Why it's important**: Defines HOW the system is built. Blueprint for implementation.

**Audience**: Developers, architects

---

### 20. **STANDARDIZATION_README.md** ⭐⭐⭐
**Purpose**: Data standardization documentation

**What it contains**:
- What was standardized
- How to use standardized data
- Integration instructions

**Why it's important**: Explains the cleaned datasets and how to use them.

**Audience**: Data analysts, developers

---

### 21. **STANDARDIZATION_SUMMARY.txt** ⭐⭐
**Purpose**: Quick summary of standardization results

**What it contains**:
- Before/after statistics
- Duplicates removed
- Files created

**Why it's important**: Quick reference for what changed.

**Audience**: Data analysts

---

## 🧪 TESTING FILES

### 22. **test_data_loader.py** ⭐⭐⭐
**Purpose**: Tests for data_loader.py

**What it tests**:
- File loading works
- Error handling works
- DataFrame structure is correct

**Why it's important**: Ensures data loading doesn't break.

---

### 23. **test_suitability_calculator.py** ⭐⭐⭐⭐
**Purpose**: Tests for suitability_calculator.py

**What it tests**:
- Score calculation is correct
- Edge cases handled
- Formula produces expected results

**Why it's important**: Core algorithm must be correct - this validates it.

---

### 24. **test_medicine_generator.py** ⭐⭐⭐
**Purpose**: Tests for medicine_generator.py

**What it tests**:
- Generates correct number of medicines
- Distributions are realistic
- No invalid values

**Why it's important**: Training data quality affects model accuracy.

---

### 25. **test_interaction_dataset.py** ⭐⭐⭐
**Purpose**: Tests interaction dataset creation

**What it tests**:
- All persona-medicine pairs created
- Features extracted correctly
- No missing values

**Why it's important**: Training dataset must be complete and correct.

---

## 🔧 UTILITY FILES

### 26. **utils.py** ⭐⭐⭐
**Purpose**: Helper functions used across modules

**What it contains**:
- Common calculations
- Data transformations
- Utility functions

**Why it's important**: Reduces code duplication, centralizes common logic.

---

### 27. **generate_personas.py** ⭐⭐⭐⭐
**Purpose**: Generates the 10,000 persona dataset

**What it does**:
- Creates realistic demographic distributions
- Assigns diseases based on age/income
- Generates behavioral attributes
- Saves to `indian_healthcare_personas.json`

**Why it's important**: Creates the foundation dataset. Run once to generate personas.

---

### 28. **check_duplicates.py** ⭐⭐
**Purpose**: Checks for duplicate records in datasets

**What it does**:
- Scans CSV files for duplicates
- Reports duplicate counts
- Helps with data quality

**Why it's important**: Data quality tool - ensures clean data.

---

### 29. **standardize_dataset.py** ⭐⭐⭐⭐
**Purpose**: Standardizes archive datasets

**What it does**:
- Cleans column names
- Standardizes dates
- Removes duplicates
- Fixes data types
- Saves to `standardized_data/`

**Why it's important**: Transforms messy real-world data into clean, usable format.

**Impact**: Removed 914 duplicates, standardized 30 columns across 4 datasets.

---

### 30. **generate_standardization_report.py** ⭐⭐
**Purpose**: Generates detailed standardization report

**What it does**:
- Compares before/after statistics
- Shows what changed
- Documents improvements

**Why it's important**: Transparency - shows exactly what standardization did.

---

### 31. **validate_standardized_data.py** ⭐⭐
**Purpose**: Validates standardized datasets

**What it does**:
- Checks column naming conventions
- Verifies no duplicates
- Confirms data types
- Previews sample data

**Why it's important**: Quality assurance - ensures standardization worked correctly.

---

### 32. **verify_*.py** files ⭐⭐
**Purpose**: Verification scripts for specific modules

**Files**:
- `verify_interaction_dataset.py`
- `verify_medicine_generator.py`
- `verify_suitability_calculator.py`

**What they do**: Quick checks that modules work correctly

**Why they're important**: Fast validation during development.

---

### 33. **example_usage.py** ⭐⭐⭐
**Purpose**: Example code showing how to use the system

**What it contains**:
- Sample code for loading data
- Example predictions
- Usage patterns

**Why it's important**: Helps developers understand how to use the modules programmatically.

---

## 🖼️ MEDIA FILES

### 34. **pharma.png** ⭐⭐
**Purpose**: PharmaDrishti logo

**What it is**: Logo image displayed in dashboard

**Why it's important**: Branding - makes the dashboard look professional.

---

## ⚙️ CONFIGURATION FILES

### 35. **requirements.txt** ⭐⭐⭐⭐⭐
**Purpose**: Python dependencies list

**What it contains**:
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.14.0
xgboost>=1.7.0
scikit-learn>=1.3.0
google-generativeai>=0.3.0
```

**Why it's important**: CRITICAL - without this, you can't install dependencies. First file to use after cloning.

**Usage**: `pip install -r requirements.txt`

---

### 36. **run_dashboard.bat** ⭐⭐⭐
**Purpose**: Windows batch script to start dashboard

**What it does**:
```batch
streamlit run dashboard.py
```

**Why it's important**: One-click dashboard launch for Windows users.

---

### 37. **retrain_model.bat** ⭐⭐
**Purpose**: Windows batch script to retrain model

**What it does**:
```batch
python train_model.py
```

**Why it's important**: One-click model retraining for Windows users.

---

## 📦 SYSTEM FILES

### 38. **.DS_Store** ⭐
**Purpose**: macOS system file (metadata)

**What it is**: Automatically created by macOS Finder

**Why it's there**: System file, can be ignored/deleted

**Note**: Should be in `.gitignore`

---

### 39. **archive.zip** ⭐⭐
**Purpose**: Compressed backup of archive folder

**What it contains**: Zipped version of all CSV files

**Why it's important**: Backup and easy distribution of datasets.

---

### 40. **.git/** folder ⭐⭐⭐⭐
**Purpose**: Git version control data

**What it contains**:
- Commit history
- Branches
- Configuration

**Why it's important**: Tracks all changes, enables collaboration, allows rollback.

---

## 📊 SUMMARY BY IMPORTANCE

### ⭐⭐⭐⭐⭐ CRITICAL (Cannot run without these)
1. `dashboard.py` - Main application
2. `suitability_calculator.py` - Core algorithm
3. `adoption_model.pkl` - Trained AI model
4. `indian_healthcare_personas.json` - Persona data
5. `requirements.txt` - Dependencies
6. `README.md` - Project overview
7. `train_model.py` - Model training

### ⭐⭐⭐⭐ VERY IMPORTANT (Needed for full functionality)
8. `data_loader.py` - Data loading
9. `standardize_dataset.py` - Data cleaning
10. `standardized_data/` - Clean datasets
11. `DASHBOARD_GUIDE.md` - User manual
12. `FEATURES.md` - Feature docs
13. `requirements.md` - Requirements
14. `design.md` - Technical design

### ⭐⭐⭐ IMPORTANT (Enhances functionality)
15. `medicine_generator.py` - Training data generation
16. `feature_names.pkl` - Feature consistency
17. `archive/` - Original datasets
18. `generate_personas.py` - Persona generation
19. `test_*.py` - Testing files
20. `utils.py` - Helper functions

### ⭐⭐ USEFUL (Nice to have)
21. Documentation files (QUICK_REFERENCE, STANDARDIZATION_SUMMARY)
22. Verification scripts
23. Batch files (.bat)
24. `pharma.png` - Logo

### ⭐ OPTIONAL (Can be ignored)
25. `.DS_Store` - System file
26. `archive.zip` - Backup

---

## 🎯 QUICK START PRIORITY

**To run PharmaDrishti, you need (in order)**:

1. ✅ Install Python 3.8+
2. ✅ Install dependencies: `pip install -r requirements.txt`
3. ✅ Ensure `indian_healthcare_personas.json` exists in `data/`
4. ✅ Ensure `adoption_model.pkl` exists (or run `train_model.py`)
5. ✅ Run dashboard: `streamlit run dashboard.py`

**Optional but recommended**:
- Configure Gemini API key for AI insights
- Read `DASHBOARD_GUIDE.md` to understand features
- Review `FEATURES.md` for detailed explanations

---

## 📚 READING ORDER FOR NEW USERS

1. **README.md** - Understand what PharmaDrishti is
2. **DASHBOARD_SETUP.md** - Get it running
3. **DASHBOARD_GUIDE.md** - Learn how to use it
4. **FEATURES.md** - Deep dive into features
5. **METRICS_AND_FORMULAS.md** - Understand the math
6. **TECHNICAL_ANALYSIS.md** - Understand the architecture

---

**Total Files**: 40+
**Total Lines of Code**: ~3,000+
**Total Documentation**: ~15,000+ words
**Data Size**: ~50MB (models + data)

