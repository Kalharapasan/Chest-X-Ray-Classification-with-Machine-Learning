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

This project implements machine learning algorithms to classify chest X-ray images into two categories:
- **NORMAL**: Healthy chest X-rays
- **PNEUMONIA**: X-rays showing signs of pneumonia

The project uses the Chest X-Ray Images (Pneumonia) dataset from Kaggle and implements multiple classification approaches to compare their effectiveness.

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

**Total Images**: 5,856 chest X-ray images
- **Training Set**: 5,216 images
- **Test Set**: 624 images  
- **Validation Set**: 16 images

## 🛠 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd "Chest X-Ray Classification with Machine Learning"
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Kaggle API** (for dataset download):
   - Go to [Kaggle Account Settings](https://www.kaggle.com/account)
   - Create new API token (downloads `kaggle.json`)
   - Place `kaggle.json` in `~/.kaggle/` directory

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

## 🤖 Models

The project implements and compares several machine learning approaches:

1. **Decision Tree Classifier**
   - Simple, interpretable tree-based model
   - Good for understanding feature importance

2. **Random Forest Classifier** 
   - Ensemble method using multiple decision trees
   - Reduces overfitting and improves accuracy

3. **Deep Learning Models** (TensorFlow/Keras)
   - Convolutional Neural Networks (CNNs)
   - Advanced feature extraction for medical images

4. **Additional ML Algorithms**
   - Various other classification techniques for comparison

## 📦 Requirements

- Python 3.7+
- numpy >= 1.21
- pandas >= 1.3
- matplotlib >= 3.5
- seaborn >= 0.11
- opencv-python >= 4.5
- tensorflow >= 2.8
- jupyter
- notebook
- scikit-learn
- kaggle (for dataset download)

## 📈 Results

The notebook provides comprehensive analysis including:
- Model accuracy comparisons
- Confusion matrices
- ROC curves and AUC scores
- Feature importance analysis
- Visualization of correctly/incorrectly classified images

*(Detailed results will be available after running the complete notebook)*

## 🤝 Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 [License](./LICENSE.md): Proprietary – Permission Required

This project is for educational purposes. Please respect the original dataset license and Kaggle terms of service.

## 🙏 Acknowledgments

- Dataset provided by Paul Mooney on Kaggle
- Original research and data collection by various medical institutions
- TensorFlow and scikit-learn communities for excellent ML libraries" 
