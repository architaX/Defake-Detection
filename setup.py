import os
from pathlib import Path

def create_project_structure():
    # Define the project root directory
    project_root = Path("fakedetect")
    
    # Define the directory structure
    structure = [
        # Root files
        project_root / "README.md",
        project_root / ".gitignore",
        project_root / ".env",
        
        # Backend structure
        project_root / "backend" / "app.py",
        project_root / "backend" / "config.py",
        project_root / "backend" / "requirements.txt",
        project_root / "backend" / "models" / "__init__.py",
        project_root / "backend" / "utils" / "preprocessing.py",
        project_root / "backend" / "utils" / "model_loader.py",
        project_root / "backend" / "utils" / "helpers.py",
        project_root / "backend" / "static" / "uploads" / ".gitkeep",
        project_root / "backend" / "static" / "css" / "style.css",
        project_root / "backend" / "static" / "js" / "script.js",
        
        # Frontend structure
        project_root / "frontend" / "index.html",
        project_root / "frontend" / "css" / "style.css",
        project_root / "frontend" / "js" / "script.js",
        
        # Training structure
        project_root / "training" / "config.py",
        project_root / "training" / "train_image.py",
        project_root / "training" / "train_audio.py",
        project_root / "training" / "train_video.py",
        
        # Tests structure
        project_root / "tests" / "test_api.py",
        project_root / "tests" / "test_models.py",
        project_root / "tests" / "test_data" / "test_image.jpg",
        project_root / "tests" / "test_data" / "test_audio.wav",
        project_root / "tests" / "test_data" / "test_video.mp4"
    ]

    # Create directories and empty files
    for item in structure:
        if item.suffix:  # It's a file
            item.parent.mkdir(parents=True, exist_ok=True)
            print(f"Creating file: {item}")
            item.touch()
        else:  # It's a directory
            print(f"Creating directory: {item}")
            item.mkdir(parents=True, exist_ok=True)

    print("\nProject structure created successfully!")

if __name__ == "__main__":
    create_project_structure()