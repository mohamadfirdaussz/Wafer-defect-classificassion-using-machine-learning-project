#!/usr/bin/env python3
"""
Cross-Platform Setup Script for WM-811K Wafer Defect Classification Project
This script automatically sets up the environment on Windows, Linux, and macOS.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(message):
    """Print a formatted header message"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 70}")
    print(f"{message:^70}")
    print(f"{'=' * 70}{Colors.ENDC}\n")


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


def check_python_version():
    """Check if Python version is 3.9 or higher"""
    print_info("Checking Python version...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print_error(f"Python 3.9+ is required. You have Python {version.major}.{version.minor}.{version.micro}")
        print_info("Please upgrade Python from: https://www.python.org/downloads/")
        return False
    
    print_success(f"Python {version.major}.{version.minor}.{version.micro} detected")
    return True


def create_virtual_environment(venv_path):
    """Create a virtual environment"""
    print_info("Creating virtual environment...")
    
    if venv_path.exists():
        print_warning("Virtual environment already exists. Skipping creation.")
        return True
    
    try:
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
        print_success("Virtual environment created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to create virtual environment: {e}")
        return False


def get_pip_executable(venv_path):
    """Get the pip executable path based on the operating system"""
    system = platform.system()
    
    if system == "Windows":
        return venv_path / "Scripts" / "pip.exe"
    else:  # Linux, macOS
        return venv_path / "bin" / "pip"


def get_python_executable(venv_path):
    """Get the Python executable path based on the operating system"""
    system = platform.system()
    
    if system == "Windows":
        return venv_path / "Scripts" / "python.exe"
    else:  # Linux, macOS
        return venv_path / "bin" / "python"


def upgrade_pip(pip_executable):
    """Upgrade pip, setuptools, and wheel"""
    print_info("Upgrading pip, setuptools, and wheel...")
    
    try:
        subprocess.run(
            [str(pip_executable), "install", "--upgrade", "pip", "setuptools", "wheel", "--quiet"],
            check=True
        )
        print_success("Pip upgraded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to upgrade pip: {e}")
        return False


def install_dependencies(pip_executable, requirements_file):
    """Install dependencies from requirements file"""
    print_info("Installing dependencies (this may take several minutes)...")
    
    try:
        subprocess.run(
            [str(pip_executable), "install", "-r", str(requirements_file)],
            check=True
        )
        print_success("All dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {e}")
        print_info(f"Try manually running:\n  {pip_executable} install -r {requirements_file}")
        return False


def check_dataset(project_root):
    """Check if dataset exists"""
    print_info("Checking dataset...")
    
    datasets_dir = project_root / "datasets"
    dataset_file = datasets_dir / "LSWMD.pkl"
    
    # Create datasets directory if it doesn't exist
    if not datasets_dir.exists():
        datasets_dir.mkdir(parents=True, exist_ok=True)
        print_info("Created datasets folder")
    
    if not dataset_file.exists():
        print("\n" + "┌" + "─" * 68 + "┐")
        print(f"│{'⚠️  DATASET NOT FOUND':^68}│")
        print("└" + "─" * 68 + "┘\n")
        print("  The wafer dataset (LSWMD.pkl) is required to run the pipeline.")
        print("\n  📥 Download from Kaggle:")
        print("  https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map")
        print(f"\n  📂 Place the file here:")
        print(f"  {dataset_file}")
        print("\n  Once downloaded, you can run the pipeline using:")
        print("  python ml_flow/main.py")
        return False
    else:
        print_success(f"Dataset found: {dataset_file}")
        return True


def get_activation_command(venv_path):
    """Get the activation command based on the operating system"""
    system = platform.system()
    
    if system == "Windows":
        return f".venv\\Scripts\\activate"
    else:  # Linux, macOS
        return f"source .venv/bin/activate"


def main():
    """Main setup function"""
    print_header("WM-811K WAFER DEFECT CLASSIFICATION - SETUP")
    
    # Get project root directory
    project_root = Path(__file__).parent.resolve()
    venv_path = project_root / ".venv"
    requirements_file = project_root / "requirement.txt"
    
    print_info(f"Project directory: {project_root}")
    print_info(f"Operating system: {platform.system()}")
    print()
    
    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)
    print()
    
    # Step 2: Create virtual environment
    if not create_virtual_environment(venv_path):
        sys.exit(1)
    print()
    
    # Step 3: Get pip and python executables
    pip_executable = get_pip_executable(venv_path)
    python_executable = get_python_executable(venv_path)
    
    # Step 4: Upgrade pip
    if not upgrade_pip(pip_executable):
        sys.exit(1)
    print()
    
    # Step 5: Install dependencies
    if not requirements_file.exists():
        print_error(f"Requirements file not found: {requirements_file}")
        sys.exit(1)
    
    if not install_dependencies(pip_executable, requirements_file):
        sys.exit(1)
    print()
    
    # Step 6: Check dataset
    dataset_exists = check_dataset(project_root)
    print()
    
    # Final message
    activation_cmd = get_activation_command(venv_path)
    
    if dataset_exists:
        print_header("✓ SETUP COMPLETE - READY TO RUN!")
        print("  Your environment is ready. To run the pipeline:\n")
        print("  1. Activate the environment:")
        print(f"     {activation_cmd}\n")
        print("  2. Run the pipeline:")
        print("     python ml_flow/main.py\n")
    else:
        print_header("SETUP COMPLETE - DATASET REQUIRED")
        print("  Environment setup is complete.")
        print("  Please download the dataset and run the pipeline.\n")
    
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
