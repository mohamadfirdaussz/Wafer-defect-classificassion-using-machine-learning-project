# 🚀 Quick Start Guide - WM-811K Wafer Defect Classification

This guide will help you get the project running on your computer in minutes.

## Prerequisites

Before you begin, ensure you have:

- ✅ **Python 3.9 or higher** installed ([Download here](https://www.python.org/downloads/))
- ✅ **8GB+ RAM** (recommended for processing large datasets)
- ✅ **5GB+ free disk space** (for dataset and results)

## Installation

### Windows Users (Recommended Method)

1. **Run Setup**
   - Double-click `setup.bat`
   - The script will automatically:
     - Check your Python version
     - Create a virtual environment
     - Install all dependencies
     - Check for the dataset

2. **Download Dataset** (if not included)
   
   If setup indicates the dataset is missing:
   - Download `LSWMD.pkl` from [Kaggle](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)
   - Create a `datasets` folder in the project directory
   - Place `LSWMD.pkl` inside: `datasets/LSWMD.pkl`

3. **Run the Pipeline**
   - Double-click `run_pipeline.bat`
   - Wait for completion (20-60 minutes depending on your hardware)

### Cross-Platform Method (Windows/Linux/macOS)

1. **Run Setup Script**
   ```bash
   python setup.py
   ```

2. **Download Dataset** (if not included)
   
   Download from [Kaggle](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map) and place in `datasets/LSWMD.pkl`

3. **Run the Pipeline**
   ```bash
   # Activate virtual environment
   # Windows:
   .venv\Scripts\activate
   
   # Linux/macOS:
   source .venv/bin/activate
   
   # Run pipeline
   python ml_flow/main.py
   ```

### Manual Installation (Advanced Users)

If you prefer manual control:

```bash
# 1. Create virtual environment
python -m venv .venv

# 2. Activate environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install dependencies
pip install -r requirement.txt

# 5. Ensure dataset is in place
# datasets/LSWMD.pkl should exist

# 6. Run pipeline
python ml_flow/main.py
```

## Expected Execution Time

| Stage | Description | Estimated Time |
|-------|-------------|----------------|
| 1 | Data Loading & Cleaning | 5-10 minutes |
| 2 | Feature Engineering | 5-10 minutes |
| 3 | Preprocessing & Balancing | 5-10 minutes |
| 3.5 | Feature Expansion | 2-5 minutes |
| 4 | Feature Selection | 10-20 minutes |
| 5 | Model Training & Tuning | 10-30 minutes |

**Total: ~40-85 minutes** (varies based on hardware)

## Checking Results

After successful completion, check these directories:

```
📁 data_loader_results/          # Cleaned wafer maps
📁 Feature_engineering_results/   # Extracted features (66 features)
📁 preprocessing_results/         # Balanced datasets
📁 feature_selection_results/     # Selected features (3 tracks)
📁 model_artifacts/              # 🏆 Final models and results
   └── master_model_comparison.csv   # Performance leaderboard
```

**Key File to Review:** `model_artifacts/master_model_comparison.csv` contains the final performance comparison of all models.

## Troubleshooting

### Python Version Error

**Error:** "Python 3.9+ is required"

**Solution:**
- Upgrade Python from [python.org](https://www.python.org/downloads/)
- During installation, check "Add Python to PATH"

### Virtual Environment Issues

**Error:** "Failed to create virtual environment"

**Solution:**
```bash
# Ensure venv module is installed
python -m pip install --upgrade pip
python -m ensurepip

# Try creating environment again
python -m venv .venv
```

### Dependency Installation Fails

**Error:** Package installation errors

**Solution:**
```bash
# Activate environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Upgrade pip and try again
python -m pip install --upgrade pip setuptools wheel
pip install -r requirement.txt
```

### Dataset Not Found

**Error:** "Dataset not found"

**Solution:**
1. Download from [Kaggle WM-811K Dataset](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)
2. Create `datasets` folder in project root
3. Place file as: `datasets/LSWMD.pkl`

### Out of Memory

**Error:** Process killed or memory error

**Solution:**
- Close other applications
- Ensure you have 8GB+ RAM available
- If still failing, reduce `N_PREFILTER` in `ml_flow/config.py`

### Pipeline Fails Mid-Execution

**Error:** Pipeline stops at a specific stage

**Solution:**
1. Check `pipeline.log` for detailed error messages
2. Try running stages individually:
   ```bash
   cd ml_flow
   python data_loader.py       # Stage 1
   python feature_engineering.py  # Stage 2
   python data_preprocessor.py    # Stage 3
   python feature_combination.py  # Stage 3.5
   python feature_selection.py    # Stage 4
   python model_tuning.py         # Stage 5
   ```

## Running Individual Stages

For debugging or partial execution:

```bash
cd ml_flow

# Run only data loading
python data_loader.py

# Run only feature engineering
python feature_engineering.py

# Run only preprocessing
python data_preprocessor.py

# Run only feature expansion
python feature_combination.py

# Run only feature selection
python feature_selection.py

# Run only model training
python model_tuning.py
```

## Testing the Installation

Verify your setup is working:

```bash
# Activate environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Run unit tests
cd ml_flow
python unit_test.py
```

Expected output:
```
......
----------------------------------------------------------------------
Ran 6 tests in 2.5s

OK
```

## Next Steps

After successful setup and execution:

1. **Review Results**
   - Open `model_artifacts/master_model_comparison.csv`
   - Check confusion matrices and ROC curves in `model_artifacts/`

2. **Try the Dashboard** (Optional)
   ```bash
   cd dashboard
   python dashboard_server.py
   ```
   Open browser to `http://localhost:5000`

3. **Read Full Documentation**
   - See `README.md` for complete project details
   - Check `project_flowchart.md` for pipeline architecture

## Getting Help

If you encounter issues not covered here:

1. Check `pipeline.log` for detailed error messages
2. Review `README.md` for detailed documentation
3. Run unit tests to verify installation
4. Open an issue on GitHub with:
   - Error message
   - Python version
   - Operating system
   - Steps you've tried

---

**Ready to start?** Run `setup.bat` (Windows) or `python setup.py` (Cross-platform) now! 🚀
