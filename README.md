"# Chest X-Ray Classification with Machine Learning

A comprehensive machine learning project for classifying chest X-rays to detect pneumonia using various ML algorithms including Decision Trees, Random Forest, and deep learning approaches.

## 📋 Table of Contents
- [📋 Table of Contents](#-table-of-contents)
- [🔍 Overview](#-overview)
- [📊 Dataset](#-dataset)
- [🛠 Installation](#-installation)
- [🚀 Usage](#-usage)
  - [Option 1: Download Dataset via Kaggle API](#option-1-download-dataset-via-kaggle-api)
  - [Option 2: Manual Dataset Setup](#option-2-manual-dataset-setup)
  - [Running the Project](#running-the-project)
- [📁 Project Structure](#-project-structure)
- [🤖 Models](#-models)
- [📦 Requirements](#-requirements)
- [📈 Results](#-results)
- [🤝 Contributing](#-contributing)
- [📝 License: Proprietary – Permission Required](#-license-proprietary--permission-required)
- [🙏 Acknowledgments](#-acknowledgments)

## 🔍 Overview

This project implements advanced machine learning algorithms to classify chest X-ray images for pneumonia detection, addressing a critical need in medical imaging diagnostics. The system can distinguish between:

- **NORMAL**: Healthy chest X-rays showing clear lungs
- **PNEUMONIA**: X-rays showing signs of pneumonia (bacterial, viral, or other)

### 🎯 Project Goals
- **Medical Impact**: Assist radiologists in rapid pneumonia screening
- **Technical Excellence**: Compare multiple ML approaches for medical image classification
- **Feature Engineering**: Extract meaningful patterns from chest X-ray images
- **Performance Analysis**: Achieve high accuracy while minimizing false negatives

### 🧬 Medical Context
Pneumonia is a leading cause of death worldwide, particularly affecting children and elderly patients. Early detection through chest X-ray analysis can significantly improve patient outcomes. This project aims to provide automated assistance for medical professionals in resource-constrained environments.

The project uses the Chest X-Ray Images (Pneumonia) dataset from Kaggle and implements multiple classification approaches with advanced feature extraction techniques including Gray-Level Co-occurrence Matrix (GLCM), Wavelet transforms, and deep learning methods.

## 📊 Dataset

**Source**: [Chest X-Ray Images (Pneumonia) - Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

**Dataset Structure**:
```
chest_xray/
├── train/
│   ├── NORMAL/     (1,341 images)
│   └── PNEUMONIA/  (3,875 images)
├── test/
│   ├── NORMAL/     (234 images)
│   └── PNEUMONIA/  (390 images)
└── val/
    ├── NORMAL/     (8 images)
    └── PNEUMONIA/  (8 images)
```

**Total Images**: 5,856 chest X-ray images from pediatric patients (1-5 years old)
- **Training Set**: 5,216 images (1,341 Normal + 3,875 Pneumonia)
- **Test Set**: 624 images (234 Normal + 390 Pneumonia)  
- **Validation Set**: 16 images (8 Normal + 8 Pneumonia)

### 📋 Dataset Characteristics
- **Image Format**: JPEG format chest X-rays
- **Patient Demographics**: Pediatric patients aged 1-5 years
- **Image Quality**: Various resolutions and contrast levels
- **Class Imbalance**: ~3:1 ratio (Pneumonia:Normal) requiring balancing techniques
- **Medical Validation**: All images reviewed by expert physicians
- **Pneumonia Types**: Includes bacterial and viral pneumonia cases

### 🩺 Clinical Relevance
- Images sourced from Guangzhou Women and Children's Medical Center
- Quality control performed by expert chest radiologists
- Represents real-world clinical conditions and imaging variations

## 🛠 Installation

### 💻 System Requirements
- **Operating System**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: Version 3.8 - 3.11 (recommended: 3.9)
- **Memory**: Minimum 8GB RAM (16GB recommended for full dataset)
- **Storage**: 15GB free space (5GB for dataset + processing space)
- **GPU**: Optional but recommended (CUDA-compatible for TensorFlow acceleration)

### 📦 Quick Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd "Chest X-Ray Classification with Machine Learning"
   ```

2. **Create virtual environment** (strongly recommended):
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate
   
   # macOS/Linux  
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Upgrade pip and install dependencies**:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python -c "import tensorflow, sklearn, cv2; print('All packages installed successfully!')"
   ```

### 🔑 Kaggle API Setup (for automatic dataset download)

1. **Create Kaggle account**: [kaggle.com/account/login](https://www.kaggle.com/account/login)
2. **Generate API token**:
   - Go to [Kaggle Account Settings](https://www.kaggle.com/account)
   - Scroll to "API" section → Click "Create New API Token"
   - Downloads `kaggle.json` file
3. **Install credentials**:
   ```bash
   # Windows
   mkdir %USERPROFILE%\.kaggle
   move kaggle.json %USERPROFILE%\.kaggle\
   
   # macOS/Linux
   mkdir -p ~/.kaggle
   mv kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

## 🚀 Usage

### Option 1: Download Dataset via Kaggle API
Run the Jupyter notebook which includes automatic dataset download:
```bash
jupyter notebook "Chest X-Ray Classification with Machine Learning I.ipynb"
```

### Option 2: Manual Dataset Setup
1. Download dataset from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
2. Extract to `dataset/` folder
3. Run the notebook

### Running the Project
1. Open the Jupyter notebook
2. Execute cells sequentially
3. The notebook will:
   - Download and prepare the dataset
   - Perform data preprocessing
   - Train multiple ML models
   - Evaluate model performance
   - Generate visualizations and results

## 📁 Project Structure

```
Chest X-Ray Classification with Machine Learning/
├── Chest X-Ray Classification with Machine Learning I.ipynb  # Main notebook
├── README.md                                                 # Project documentation
├── requirements.txt                                          # Python dependencies
├── dataset/                                                  # Dataset folder
│   └── chest_xray/
│       ├── train/
│       ├── test/
│       └── val/
└── __MACOSX/                                                # macOS metadata (ignore)
```

## 🤖 Models & Feature Engineering

### 🔬 Advanced Feature Extraction Pipeline

#### 1. **Gray-Level Co-occurrence Matrix (GLCM) Features**
   - **Purpose**: Captures texture patterns in chest X-rays
   - **Features Extracted**:
     - Dissimilarity: Measures local variation
     - Correlation: Linear dependency of gray levels
     - Homogeneity: Closeness of distribution elements
     - Energy: Uniformity of gray-level distribution
   - **Configuration**: Multi-directional analysis (0°, 45°, 90°, 135°)

#### 2. **Wavelet Transform Features**
   - **Purpose**: Multi-resolution analysis for detecting abnormalities
   - **Decomposition**: Haar wavelet with level-1 decomposition
   - **Features**:
     - LH Energy: Horizontal edge information
     - HL Energy: Vertical edge information  
     - HH Energy: Diagonal edge information
   - **Medical Relevance**: Detects fine structural changes in lung tissue

#### 3. **Intensity-Based Features**
   - **Statistical Measures**: Mean and standard deviation
   - **Purpose**: Captures overall brightness and contrast patterns
   - **Clinical Significance**: Pneumonia often alters lung opacity

#### 4. **Advanced Embedding Technique**
   - **Block-Based Processing**: Images divided into 4×4 grids
   - **Binary Encoding**: 16-bit depth feature encoding
   - **Multi-Scale Analysis**: Captures both local and global patterns

### 🧠 Machine Learning Models

#### 1. **Decision Tree Classifier**
   - **Advantages**: Highly interpretable, medical professionals can understand decision paths
   - **Feature Selection**: Automatic identification of most discriminative features
   - **Medical Application**: Provides clear diagnostic rules

#### 2. **Random Forest Classifier**
   - **Ensemble Approach**: Combines 100+ decision trees
   - **Robustness**: Reduces overfitting common in medical imaging
   - **Feature Importance**: Ranks which image characteristics are most diagnostic
   - **Performance**: Generally superior to single decision trees

#### 3. **Support Vector Machine (SVM)**
   - **Kernel Methods**: RBF and polynomial kernels for complex patterns
   - **High-Dimensional Data**: Effective with extracted feature vectors
   - **Medical Imaging**: Proven effectiveness in radiological applications

#### 4. **Deep Learning Models** (TensorFlow/Keras)
   - **Convolutional Neural Networks (CNNs)**:
     - Custom architectures optimized for chest X-rays
     - Transfer learning from pre-trained medical imaging models
     - Data augmentation for improved generalization
   - **Hybrid Approaches**: Combining traditional features with deep learning

#### 5. **Additional Algorithms**
   - **Logistic Regression**: Baseline linear classifier
   - **K-Nearest Neighbors (KNN)**: Instance-based learning
   - **Gradient Boosting**: Advanced ensemble methods
   - **Neural Networks**: Multi-layer perceptrons

## 📦 Requirements

### 🐍 Core Dependencies

**Essential Packages:**
```
Python >= 3.8, <= 3.11
numpy >= 1.21.0, < 1.25.0
pandas >= 1.3.0
matplotlib >= 3.5.0
seaborn >= 0.11.0
opencv-python >= 4.5.0
scikit-learn >= 1.1.0
scipy >= 1.7.0
```

**Deep Learning:**
```
tensorflow >= 2.8.0, < 2.14.0  # GPU support recommended
keras >= 2.8.0
```

**Image Processing:**
```
Pillow >= 8.0.0
scikit-image >= 0.19.0
PyWavelets >= 1.3.0  # For wavelet transforms
```

**Development & Analysis:**
```
jupyter >= 1.0.0
notebook >= 6.4.0
ipywidgets >= 7.6.0  # For interactive widgets
kaggle >= 1.5.12     # Dataset download
```

### 🚀 Optional GPU Support

**For NVIDIA GPUs:**
```bash
# Install CUDA Toolkit 11.2 or 11.8
# Install cuDNN 8.1.0 or higher
pip install tensorflow-gpu  # If using older TensorFlow
```

**Verification:**
```python
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))
```

### 💾 Memory Requirements
- **Minimum**: 8GB RAM
- **Recommended**: 16GB RAM
- **Full Dataset Processing**: 32GB RAM (for simultaneous feature extraction)
- **Storage**: 15GB free space

## 📈 Results & Performance Metrics

### 🎯 Key Performance Indicators

The project evaluates models using multiple metrics critical for medical diagnosis:

#### **Primary Metrics:**
- **Accuracy**: Overall classification correctness
- **Sensitivity (Recall)**: True Positive Rate - crucial for detecting pneumonia
- **Specificity**: True Negative Rate - avoiding false alarms
- **Precision**: Positive Predictive Value
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area Under Receiver Operating Characteristic Curve

#### **Medical Significance:**
- **False Negative Rate**: Critical to minimize (missing pneumonia cases)
- **False Positive Rate**: Important to control (unnecessary treatments)
- **NPV (Negative Predictive Value)**: Confidence in normal diagnoses

### 📊 Visualization Suite

#### **Model Comparison:**
- Side-by-side accuracy comparisons across all algorithms
- Training vs. validation performance curves
- Computational efficiency analysis (training time vs. accuracy)

#### **Diagnostic Analysis:**
- **Confusion Matrices**: Detailed breakdown of predictions vs. actual
- **ROC Curves**: Sensitivity vs. Specificity trade-offs
- **Precision-Recall Curves**: Performance across different thresholds
- **Feature Importance Plots**: Which image characteristics drive decisions

#### **Medical Interpretation:**
- **Correctly Classified Cases**: Visual examples with confidence scores
- **Misclassified Cases**: Error analysis with medical insights
- **Feature Maps**: Visualization of extracted texture and intensity patterns
- **Decision Boundaries**: How models separate normal from pneumonia cases

#### **Feature Analysis:**
- **GLCM Feature Distributions**: Texture pattern comparisons
- **Wavelet Coefficient Analysis**: Multi-resolution decomposition results
- **Correlation Heatmaps**: Feature interdependencies
- **Class Separation Plots**: How well features distinguish conditions

### 🔬 Expected Performance Ranges

*(Based on similar medical imaging studies)*

| Model Type | Expected Accuracy | Sensitivity | Specificity |
|------------|------------------|-------------|-------------|
| Decision Tree | 75-85% | 80-90% | 70-80% |
| Random Forest | 85-92% | 88-95% | 82-88% |
| SVM (RBF) | 82-90% | 85-92% | 80-87% |
| CNN (Custom) | 88-95% | 90-97% | 85-92% |
| Ensemble | 90-96% | 92-98% | 88-94% |

**Note**: Actual results depend on dataset quality, preprocessing, and hyperparameter tuning.

### 📋 Detailed Results Location

*Complete performance metrics, visualizations, and analysis will be generated when you run the notebook. Results include:*

- **Section 15+**: Model training and evaluation results
- **Saved Outputs**: Model performance charts and confusion matrices
- **Feature Analysis**: GLCM and wavelet feature effectiveness
- **Clinical Insights**: Medical interpretation of model decisions

## 🔧 Troubleshooting

### ⚠️ Common Issues & Solutions

#### **Memory Errors**
```python
# Problem: "MemoryError" during feature extraction
# Solution: Enable subset processing
USE_SUBSET = True
SUBSET_SIZE = 100  # Reduce this number
```

#### **Kaggle API Issues**
```bash
# Problem: "401 Unauthorized" error
# Solution: Verify kaggle.json placement
ls ~/.kaggle/kaggle.json  # Should exist
kaggle datasets download --help  # Test API
```

#### **TensorFlow GPU Issues**
```python
# Problem: GPU not detected
# Check CUDA/cuDNN installation
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

# Force CPU if needed
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

#### **OpenCV Import Errors**
```bash
# Problem: cv2 import fails
# Solution: Reinstall opencv-python
pip uninstall opencv-python opencv-python-headless
pip install opencv-python
```

#### **Jupyter Notebook Issues**
```bash
# Problem: Widgets not displaying
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

### 🐛 Performance Optimization

#### **For Large Datasets:**
- Enable `USE_SUBSET = True` for initial testing
- Increase `SUBSET_SIZE` gradually based on available RAM
- Use GPU acceleration for deep learning models
- Consider batch processing for feature extraction

#### **For Low-Memory Systems:**
```python
# Reduce image processing resolution
TARGET_SIZE = (128, 128)  # Instead of (256, 256)

# Process images in smaller batches
BATCH_SIZE = 10  # Reduce from default
```

## ❓ Frequently Asked Questions

### **Q: How long does the complete analysis take?**
**A:** Depends on your system and dataset size:
- **Full dataset (5,856 images)**: 2-4 hours on standard laptop
- **Subset (100 images)**: 5-10 minutes
- **With GPU acceleration**: 50-75% faster for deep learning models

### **Q: Can I use my own chest X-ray images?**
**A:** Yes! Place your images in the following structure:
```
custom_dataset/
├── NORMAL/
│   ├── image1.jpg
│   └── image2.jpg
└── PNEUMONIA/
    ├── image3.jpg
    └── image4.jpg
```
Then modify the `root_path` variable in the notebook.

### **Q: Which model performs best for medical diagnosis?**
**A:** Based on medical imaging research:
1. **Random Forest**: Best balance of accuracy and interpretability
2. **CNN**: Highest accuracy but less interpretable
3. **SVM**: Good performance with extracted features
4. **Decision Tree**: Most interpretable but lower accuracy

### **Q: Are the results clinically validated?**
**A:** This is a research/educational project. Results should NOT be used for actual medical diagnosis without proper clinical validation and regulatory approval.

### **Q: How can I improve model performance?**
**A:** Try these approaches:
- Increase dataset size with data augmentation
- Tune hyperparameters using GridSearchCV
- Implement ensemble methods
- Use pre-trained medical imaging models
- Apply advanced preprocessing techniques

## 🧪 Technical Notes

### **Feature Engineering Details:**
- **GLCM Parameters**: 32 gray levels, 4 directions, distance=1 pixel
- **Wavelet Transform**: Haar wavelet, level-1 decomposition
- **Image Preprocessing**: Grayscale conversion, normalization
- **Block Processing**: 4×4 grid analysis for spatial features

### **Model Configuration:**
- **Random Forest**: 100 estimators, max_depth=10
- **Decision Tree**: Gini criterion, max_depth=5
- **SVM**: RBF kernel, C=1.0, gamma='scale'
- **Neural Network**: 3 hidden layers, ReLU activation

### **Cross-Validation:**
- **Strategy**: Stratified 5-fold cross-validation
- **Metrics**: All models evaluated with same CV strategy
- **Reproducibility**: Random seed set to 42

## 🤝 Contributing

### **How to Contribute:**

1. **Fork the project**
2. **Create feature branch**: `git checkout -b feature/AmazingFeature`
3. **Make changes**: Add new models, improve features, fix bugs
4. **Test thoroughly**: Ensure all notebooks run successfully
5. **Commit changes**: `git commit -m 'Add some AmazingFeature'`
6. **Push to branch**: `git push origin feature/AmazingFeature`
7. **Open Pull Request**: Provide detailed description

### **Contribution Areas:**
- **New ML Models**: Implement additional classification algorithms
- **Feature Engineering**: Add new image feature extraction methods
- **Visualization**: Improve charts and medical image displays
- **Performance**: Optimize processing speed and memory usage
- **Documentation**: Enhance README, add tutorials
- **Testing**: Add unit tests and validation scripts

### **Code Standards:**
- Follow PEP 8 Python style guidelines
- Add docstrings to all functions
- Include comments for complex medical/technical concepts
- Test on both subset and full dataset

## 📝 [License](./LICENSE.md): Proprietary – Permission Required

This project is for educational purposes. Please respect the original dataset license and Kaggle terms of service.

## 🙏 Acknowledgments

- Dataset provided by Paul Mooney on Kaggle
- Original research and data collection by various medical institutions
- TensorFlow and scikit-learn communities for excellent ML libraries" 
