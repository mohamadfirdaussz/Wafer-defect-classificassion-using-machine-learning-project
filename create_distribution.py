#!/usr/bin/env python3
"""
Distribution Package Creator for WM-811K Wafer Defect Classification Project
Creates a ready-to-distribute ZIP package with all necessary files including dataset.
"""

import os
import sys
import zipfile
import hashlib
from pathlib import Path
from datetime import datetime


class Colors:
    """ANSI color codes for terminal output"""
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    OKCYAN = '\033[96m'


def print_success(message):
    """Print a success message"""
    print(f"{Colors.OKGREEN}[✓] {message}{Colors.ENDC}")


def print_error(message):
    """Print an error message"""
    print(f"{Colors.FAIL}[✗] {message}{Colors.ENDC}")


def print_warning(message):
    """Print a warning message"""
    print(f"{Colors.WARNING}[!] {message}{Colors.ENDC}")


def print_info(message):
    """Print an info message"""
    print(f"{Colors.OKCYAN}[i] {message}{Colors.ENDC}")


def get_file_hash(filepath):
    """Calculate SHA256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def format_size(size_bytes):
    """Format bytes to human-readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def should_include(path, project_root):
    """Determine if a file/folder should be included in the distribution"""
    # Convert to relative path
    try:
        rel_path = path.relative_to(project_root)
    except ValueError:
        return False
    
    # Exclude patterns
    exclude_dirs = {
        '.venv', 'venv', '.env', 'env', 'project.venv', 'new_env',
        '.git', '.github', '.vscode', '.idea', '__pycache__',
        'data_loader_results', 'Feature_engineering_results',
        'preprocessing_results', 'feature_selection_results', 
        'model_artifacts', '.gemini', '.conda'
    }
    
    exclude_files = {
        '.gitignore', '.gitattributes', 'pipeline.log',
        'temp_features.json', 'temp_results.txt', '.DS_Store'
    }
    
    exclude_extensions = {'.pyc', '.pyo', '.log', '.zip'}
    
    # Check if any parent directory is in exclude list
    for part in rel_path.parts:
        if part in exclude_dirs:
            return False
    
    # Check filename
    if path.name in exclude_files:
        return False
    
    # Check extension
    if path.suffix in exclude_extensions:
        return False
    
    return True


def create_distribution_readme(project_root, dataset_exists):
    """Create a README for the distribution package"""
    readme_content = """# WM-811K Wafer Defect Classification - Distribution Package

## 🚀 Quick Start Guide

This package contains everything you need to run the wafer defect classification pipeline.

### Prerequisites
- **Python 3.9 or higher** installed on your computer
- **8GB+ RAM** recommended
- **Windows, Linux, or macOS**

### Installation Steps

#### Option 1: Automated Setup (Recommended)

**Windows:**
1. Extract this ZIP file to a folder
2. Double-click `setup.bat`
3. Wait for setup to complete
4. Double-click `run_pipeline.bat` to run the pipeline

**Linux/macOS:**
1. Extract this ZIP file to a folder
2. Open terminal in the extracted folder
3. Run: `python3 setup.py`
4. Run: `python ml_flow/main.py`

#### Option 2: Manual Setup

1. Extract this ZIP file
2. Open terminal/command prompt in the extracted folder
3. Create virtual environment:
   ```bash
   python -m venv .venv
   ```
4. Activate virtual environment:
   - Windows: `.venv\\Scripts\\activate`
   - Linux/Mac: `source .venv/bin/activate`
5. Install dependencies:
   ```bash
   pip install -r requirement.txt
   ```
6. Run pipeline:
   ```bash
   python ml_flow/main.py
   ```

"""

    if not dataset_exists:
        readme_content += """
### ⚠️ Dataset Required

The dataset file `LSWMD.pkl` is not included in this package due to size constraints.

**Download Instructions:**
1. Download from Kaggle: https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map
2. Create a `datasets` folder in the project root
3. Place `LSWMD.pkl` inside the `datasets` folder

"""

    readme_content += """
### 📊 Expected Results

After running the pipeline, you will find results in:
- `data_loader_results/` - Cleaned wafer maps
- `Feature_engineering_results/` - Extracted features
- `preprocessing_results/` - Preprocessed data
- `feature_selection_results/` - Selected features
- `model_artifacts/` - Trained models and performance metrics

Check `model_artifacts/master_model_comparison.csv` for the final leaderboard!

### 🔧 Troubleshooting

**Python not found:**
- Install Python 3.9+ from python.org
- Make sure to add Python to PATH during installation

**Dependencies fail to install:**
- Upgrade pip: `python -m pip install --upgrade pip`
- Try installing manually: `pip install -r requirement.txt`

**Pipeline fails to run:**
- Check that dataset is in the correct location
- Verify you have enough RAM (8GB+ recommended)
- Check `pipeline.log` for error details

### 📖 Full Documentation

See `README.md` for complete project documentation.

---

**Project:** WM-811K Wafer Defect Classification
**License:** MIT
**Generated:** """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """
"""

    readme_path = project_root / "DISTRIBUTION_README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    return readme_path


def create_distribution_package():
    """Main function to create distribution package"""
    print("\n" + "=" * 70)
    print("  CREATING DISTRIBUTION PACKAGE")
    print("=" * 70 + "\n")
    
    # Get project root
    project_root = Path(__file__).parent.resolve()
    dataset_path = project_root / "datasets" / "LSWMD.pkl"
    
    # Check for dataset
    dataset_exists = dataset_path.exists()
    
    if dataset_exists:
        dataset_size = dataset_path.stat().st_size
        print_success(f"Dataset found: {format_size(dataset_size)}")
    else:
        print_warning("Dataset not found - will create instructions for download")
    
    # Create distribution README
    print_info("Creating distribution README...")
    readme_path = create_distribution_readme(project_root, dataset_exists)
    print_success("Distribution README created")
    
    # Create ZIP filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"wafer_defect_classification_distribution_{timestamp}.zip"
    zip_path = project_root / zip_filename
    
    print_info(f"Creating package: {zip_filename}")
    print_info("This may take a few minutes...")
    
    # Create ZIP file
    total_files = 0
    total_size = 0
    
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Walk through project directory
            for root, dirs, files in os.walk(project_root):
                root_path = Path(root)
                
                # Skip if this directory should be excluded
                if not should_include(root_path, project_root):
                    continue
                
                for file in files:
                    file_path = root_path / file
                    
                    # Skip if file should be excluded
                    if not should_include(file_path, project_root):
                        continue
                    
                    # Skip the ZIP file itself
                    if file_path == zip_path:
                        continue
                    
                    # Add to ZIP
                    arcname = file_path.relative_to(project_root)
                    zipf.write(file_path, arcname)
                    
                    total_files += 1
                    total_size += file_path.stat().st_size
                    
                    # Print progress for every 10 files
                    if total_files % 10 == 0:
                        print(f"  Added {total_files} files...", end='\r')
        
        print()  # New line after progress
        
        # Get final package size
        package_size = zip_path.stat().st_size
        
        # Calculate checksum
        print_info("Calculating checksum...")
        checksum = get_file_hash(zip_path)
        
        # Success message
        print()
        print("=" * 70)
        print_success("DISTRIBUTION PACKAGE CREATED SUCCESSFULLY")
        print("=" * 70)
        print()
        print(f"  📦 Package:     {zip_filename}")
        print(f"  📊 Size:        {format_size(package_size)}")
        print(f"  📁 Files:       {total_files}")
        print(f"  🔐 SHA256:      {checksum[:16]}...")
        print(f"  📍 Location:    {zip_path}")
        print()
        
        if dataset_exists:
            print_success("Dataset included - Package is ready for complete deployment")
        else:
            print_warning("Dataset NOT included - Recipients will need to download separately")
        
        print()
        print("  Next Steps:")
        print("  1. Share this ZIP file with recipients")
        print("  2. Recipients should extract and run setup.bat (Windows) or setup.py")
        print("  3. Recipients can then run the pipeline")
        print()
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print_error(f"Failed to create package: {e}")
        return False
    finally:
        # Clean up distribution README
        if readme_path.exists():
            try:
                readme_path.unlink()
            except:
                pass


def main():
    """Main entry point"""
    try:
        success = create_distribution_package()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print()
        print_warning("Package creation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
