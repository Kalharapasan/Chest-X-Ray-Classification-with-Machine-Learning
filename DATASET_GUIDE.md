# 📥 Complete Guide: How to Get Dataset for Pneumonia Detection

## 🎯 Dataset Overview

**Dataset Name:** Chest X-Ray Images (Pneumonia)  
**Source:** Kaggle  
**Size:** ~1.15 GB  
**Total Images:** 5,856 images  
**Classes:** 2 (NORMAL, PNEUMONIA)  

### Dataset Structure:
```
chest_xray/
├── train/           (5,216 images)
│   ├── NORMAL/      (1,341 images)
│   └── PNEUMONIA/   (3,875 images)
├── test/            (624 images)
│   ├── NORMAL/      (234 images)
│   └── PNEUMONIA/   (390 images)
└── val/             (16 images)
    ├── NORMAL/      (8 images)
    └── PNEUMONIA/   (8 images)
```

---

## 🔐 Method 1: Using Kaggle API (Recommended for Jupyter/Colab)

### Step 1: Create Kaggle Account
1. Go to [https://www.kaggle.com/](https://www.kaggle.com/)
2. Click "Register" (top right)
3. Sign up with email or Google account
4. Verify your email

### Step 2: Get API Credentials
1. Click on your profile picture (top right)
2. Select "Account" from dropdown
3. Scroll down to "API" section
4. Click "Create New API Token"
5. This downloads `kaggle.json` file to your computer

### Step 3: Install Kaggle Package

**For Jupyter Notebook:**
```python
!pip install kaggle
```

**For command line:**
```bash
pip install kaggle
```

### Step 4: Setup Kaggle Credentials

**Option A: For Jupyter/Google Colab:**
```python
# Upload kaggle.json file
from google.colab import files
uploaded = files.upload()  # Select kaggle.json when prompted

# Move to correct location
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

**Option B: For Local Machine (Linux/Mac):**
```bash
# Create .kaggle directory
mkdir -p ~/.kaggle

# Move kaggle.json to .kaggle directory
mv ~/Downloads/kaggle.json ~/.kaggle/

# Set permissions
chmod 600 ~/.kaggle/kaggle.json
```

**Option C: For Windows:**
```bash
# Create directory
mkdir %USERPROFILE%\.kaggle

# Move kaggle.json (in Command Prompt)
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\

# Or in PowerShell
Move-Item "$env:USERPROFILE\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\"
```

### Step 5: Download Dataset

**In Jupyter Notebook:**
```python
# Download dataset
!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia

# Unzip
!unzip -q chest-xray-pneumonia.zip -d ./dataset

print("✅ Dataset downloaded and extracted!")
```

**In Terminal/Command Line:**
```bash
# Download
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia

# Unzip
unzip chest-xray-pneumonia.zip -d ./dataset
```

### Step 6: Verify Dataset
```python
import os

dataset_path = './dataset/chest_xray'

# Check structure
for root, dirs, files in os.walk(dataset_path):
    level = root.replace(dataset_path, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files[:3]:  # Show first 3 files
        print(f'{subindent}{file}')
    if len(files) > 3:
        print(f'{subindent}... and {len(files)-3} more files')
```

---

## 💻 Method 2: Manual Download (Easier for Beginners)

### Step 1: Visit Dataset Page
1. Go to: [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
2. You may need to sign in or create a Kaggle account

### Step 2: Download
1. Click the **"Download"** button (top right corner)
2. The file `chest-xray-pneumonia.zip` (~1.15 GB) will download
3. Wait for download to complete

### Step 3: Extract Files

**On Windows:**
1. Right-click `chest-xray-pneumonia.zip`
2. Select "Extract All..."
3. Choose destination folder
4. Click "Extract"

**On Mac:**
1. Double-click `chest-xray-pneumonia.zip`
2. Files will extract automatically

**On Linux:**
```bash
unzip chest-xray-pneumonia.zip -d ./dataset
```

### Step 4: Organize Files
Move the extracted folder to your project directory:
```
your_project/
├── dataset/
│   └── chest_xray/
│       ├── train/
│       ├── test/
│       └── val/
├── pneumonia_detection_jupyter.ipynb
└── pneumonia_detector_ui.py
```

---

## 🐍 Method 3: Direct Download with Python (No Kaggle API)

If Kaggle API doesn't work, use this alternative:

```python
import requests
import zipfile
import os
from tqdm import tqdm

def download_file(url, filename):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

# Note: You'll need to get the direct download URL from Kaggle
# This requires authentication, so Method 1 or 2 is recommended
```

---

## 📱 Method 4: Using Google Colab (Cloud-Based)

### Step 1: Open Google Colab
1. Go to [https://colab.research.google.com/](https://colab.research.google.com/)
2. Click "New Notebook"

### Step 2: Mount Google Drive (Optional)
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 3: Upload kaggle.json
```python
from google.colab import files
uploaded = files.upload()
```

### Step 4: Setup and Download
```python
# Install kaggle
!pip install -q kaggle

# Setup credentials
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download dataset
!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia

# Unzip
!unzip -q chest-xray-pneumonia.zip

# Verify
!ls chest_xray/train/
```

### Step 5: Update Path in Code
```python
# Use this path in Colab
root_path = '/content/chest_xray/train'
```

---

## 📊 Verify Your Dataset

After downloading, verify the dataset with this code:

```python
import os
import pandas as pd

def verify_dataset(root_path):
    """Verify dataset structure and count images"""
    
    print("Checking dataset structure...")
    print("="*60)
    
    # Check if path exists
    if not os.path.exists(root_path):
        print(f"❌ Error: Path does not exist: {root_path}")
        return False
    
    # Check for subdirectories
    subdirs = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    
    if len(subdirs) == 0:
        print(f"❌ Error: No subdirectories found in {root_path}")
        return False
    
    print(f"✓ Found {len(subdirs)} classes: {subdirs}")
    print()
    
    # Count images in each class
    total_images = 0
    for subdir in subdirs:
        subdir_path = os.path.join(root_path, subdir)
        images = [f for f in os.listdir(subdir_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        count = len(images)
        total_images += count
        print(f"  {subdir}: {count} images")
        
        # Show first few filenames
        if count > 0:
            print(f"    Sample files: {images[:3]}")
    
    print()
    print(f"✓ Total images: {total_images}")
    print("="*60)
    
    if total_images > 0:
        print("✅ Dataset verified successfully!")
        return True
    else:
        print("❌ No images found!")
        return False

# Run verification
root_path = './dataset/chest_xray/train'  # Adjust this path
verify_dataset(root_path)
```

---

## 🔧 Troubleshooting

### Problem 1: "kaggle.json not found"
**Solution:**
```bash
# Check if file exists
ls ~/.kaggle/

# If not, ensure you moved it correctly
# Re-download from Kaggle account settings
```

### Problem 2: "Permission denied"
**Solution:**
```bash
# Set correct permissions
chmod 600 ~/.kaggle/kaggle.json
```

### Problem 3: "Dataset not found"
**Solution:**
- Verify you're using the correct dataset name
- Check your internet connection
- Try manual download (Method 2)

### Problem 4: "Out of disk space"
**Solution:**
- Free up ~2 GB of space (dataset is 1.15 GB + extraction)
- Delete unnecessary files
- Use cloud storage (Google Drive, Colab)

### Problem 5: "Cannot unzip file"
**Solution:**
```python
# Try different unzip method
import zipfile

with zipfile.ZipFile('chest-xray-pneumonia.zip', 'r') as zip_ref:
    zip_ref.extractall('./dataset')
```

### Problem 6: Wrong path in code
**Solution:**
```python
# Find where dataset actually is
import os
import glob

# Search for chest_xray folder
for root, dirs, files in os.walk('.'):
    if 'chest_xray' in dirs:
        print(f"Found dataset at: {os.path.join(root, 'chest_xray')}")
```

---

## 📝 Dataset Information

### About the Dataset

**Source:** 
- Guangzhou Women and Children's Medical Center
- Published on Kaggle by Paul Mooney

**Citation:**
```
Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), 
"Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images 
for Classification", Mendeley Data, V2, 
doi: 10.17632/rscbjbr9sj.2
```

**Classes:**
1. **NORMAL** - Healthy chest X-rays
2. **PNEUMONIA** - X-rays showing pneumonia (bacterial and viral)

**Image Format:**
- Format: JPEG
- Color: Grayscale
- Size: Variable (will be resized in code)

**Note on Class Imbalance:**
- Training set is imbalanced (more PNEUMONIA images)
- Code includes balancing functionality
- Consider data augmentation for better results

---

## 🎯 Quick Start After Download

Once dataset is downloaded, update the path in your code:

**For Jupyter Notebook:**
```python
# Update this line in the notebook
root_path = './dataset/chest_xray/train'

# Or if using Google Colab
root_path = '/content/chest_xray/train'

# Or if using Kaggle Notebooks
root_path = '/kaggle/input/chest-xray-pneumonia/chest_xray/train'
```

**For Tkinter UI:**
Just click "Load Dataset Folder" and select:
```
your_path/chest_xray/train
```

---

## 💡 Alternative Datasets

If you have issues with this dataset, here are alternatives:

### 1. NIH Chest X-Ray Dataset
- URL: [https://www.kaggle.com/datasets/nih-chest-xrays/data](https://www.kaggle.com/datasets/nih-chest-xrays/data)
- Size: Larger, ~42 GB
- Classes: 14 different pathologies

### 2. COVID-19 Radiography Database
- URL: [https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
- Classes: COVID-19, Viral Pneumonia, Normal

### 3. RSNA Pneumonia Detection Challenge
- URL: [https://www.kaggle.com/c/rsna-pneumonia-detection-challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
- Format: DICOM files

---

## ✅ Checklist

Before running the code, ensure:

- [ ] Dataset downloaded (1.15 GB)
- [ ] Files extracted successfully
- [ ] Folder structure is correct
- [ ] Path updated in code
- [ ] Can see image files in subfolders
- [ ] At least 2 GB free disk space
- [ ] All dependencies installed

---

## 🆘 Still Having Issues?

1. **Check dataset page:** [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

2. **Read Kaggle API docs:** [https://github.com/Kaggle/kaggle-api](https://github.com/Kaggle/kaggle-api)

3. **Try the manual download method** (Method 2) - simplest approach

4. **Use Google Colab** - no local setup needed

5. **Check the discussion section** on Kaggle for help

---

**Happy Learning! 🎓🔬**
