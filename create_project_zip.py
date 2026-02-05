import zipfile
import os
from datetime import datetime

def create_zip_archive():
    # Define exclusions
    excludes = {
        '.git', '.idea', '.vscode', '.venv', 'project.venv', '.conda', '.gemini',
        '__pycache__', 'model_artifacts', 'Feature_engineering_results',
        'data_loader_results', 'feature_selection_results', 'preprocessing_results',
        'verify_FILE'
    }
    
    # Generate zip filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"wafer_project_code_{timestamp}.zip"
    
    current_dir = os.getcwd()
    
    print(f"Creating zip archive: {zip_filename}")
    print(f"Excluding: {', '.join(sorted(excludes))}")
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(current_dir):
            # Modify dirs in-place to skip excluded directories
            dirs[:] = [d for d in dirs if d not in excludes]
            
            for file in files:
                if file == zip_filename:
                    continue
                if file.endswith('.pyc') or file.endswith('.zip'):
                    continue
                    
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, current_dir)
                zipf.write(file_path, arcname)
                # print(f"Added: {arcname}")
                
    print(f"Successfully created {zip_filename}")
    return zip_filename

if __name__ == "__main__":
    create_zip_archive()
