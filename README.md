# 🩸 Blood Group Detection Using CNN

An AI-powered web application that predicts blood groups from fingerprint images using Convolutional Neural Networks (CNN).

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)

## 🔍 Overview

This project implements a deep learning system that analyzes fingerprint images to predict blood groups. The application uses a custom CNN architecture trained on fingerprint datasets, achieving an overall accuracy of **59.6%**.

## ✨ Features

- 🎯 **Real-time Blood Group Prediction** - Upload an image and get instant results
- 📊 **Interactive Visualizations** - View probability distributions with Plotly charts
- 🎨 **Modern UI** - Beautiful gradient design with responsive layout
- 📁 **Multiple Format Support** - Accepts JPG, JPEG, PNG, and BMP images
- 🔄 **Model Caching** - Fast loading with Streamlit's cache system
- 📈 **Confidence Indicators** - Visual feedback on prediction reliability
- 📱 **Responsive Design** - Works on desktop and mobile devices

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone or download the project**
```bash
cd bloodgrp_detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Train the model** (if not already trained)
```bash
python train_improved_model.py
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Open your browser** and navigate to:
```
http://localhost:8501
```

## 💻 Usage

1. **Launch the Application**
   - Run `streamlit run app.py`
   - The app will open in your default browser

2. **Upload an Image**
   - Click "Browse files" or drag-and-drop
   - Select a fingerprint image (JPG, PNG, BMP)

3. **View Results**
   - See the predicted blood group
   - Check confidence score
   - Analyze probability distribution

## 📊 Model Performance

### Overall Metrics
- **Accuracy**: 59.6%
- **Training Epochs**: 39 (with early stopping)
- **Architecture**: Custom CNN with BatchNormalization

### Per-Class Performance
| Blood Group | Accuracy | Precision | Recall |
|-------------|----------|-----------|--------|
| B-          | 91.63%   | 0.59      | 0.92   |
| A+          | 83.36%   | 0.57      | 0.83   |
| O+          | 78.99%   | 0.49      | 0.79   |
| A-          | 57.68%   | 0.48      | 0.58   |
| AB+         | 54.94%   | 0.76      | 0.55   |
| B+          | 42.33%   | 0.78      | 0.42   |
| AB-         | 40.74%   | 0.84      | 0.41   |
| O-          | 27.53%   | 0.96      | 0.28   |

### Model Architecture
```
Conv2D(32) → BatchNorm → MaxPool
Conv2D(64) → BatchNorm → MaxPool
Conv2D(128) → BatchNorm → MaxPool
Flatten
Dense(256) → Dropout(0.5)
Dense(128) → Dropout(0.3)
Dense(8, softmax)
```

## 📁 Project Structure

```
bloodgrp_detection/
├── app.py                      # Main Streamlit application
├── train_improved_model.py     # Improved model training script
├── test_model.py              # Model evaluation script
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── model/
│   ├── bloodgroup_cnn_model.keras    # Trained model
│   └── train_model.py               # Original training script
├── dataset/
│   ├── A+/                    # Blood group A+ images
│   ├── A-/                    # Blood group A- images
│   ├── B+/                    # Blood group B+ images
│   ├── B-/                    # Blood group B- images
│   ├── AB+/                   # Blood group AB+ images
│   ├── AB-/                   # Blood group AB- images
│   ├── O+/                    # Blood group O+ images
│   └── O-/                    # Blood group O- images
├── utils/
│   └── preprocess.py          # Image preprocessing utilities
└── confusion_matrix.png       # Model evaluation visualization
```

## 🛠 Technologies Used

- **Python 3.8+** - Programming language
- **TensorFlow 2.13** - Deep learning framework
- **Streamlit** - Web application framework
- **Plotly** - Interactive visualizations
- **Pillow** - Image processing
- **NumPy** - Numerical computing
- **Scikit-learn** - Model evaluation metrics
- **Matplotlib & Seaborn** - Data visualization

## 📈 Training Details

### Data Augmentation
- Rotation: ±20 degrees
- Width/Height shift: ±20%
- Horizontal flip
- Zoom: ±15%

### Optimization
- **Optimizer**: Adam (learning rate: 0.0001)
- **Loss Function**: Categorical Crossentropy
- **Callbacks**: 
  - Early Stopping (patience: 10)
  - Learning Rate Reduction (factor: 0.5, patience: 5)

### Dataset Statistics
- Total Images: 6,000
- Training Set: 4,803 images (80%)
- Validation Set: 1,197 images (20%)
- Image Size: 64x64 pixels
- Classes: 8 blood groups

## 🔧 Troubleshooting

### Model Not Found Error
```bash
python train_improved_model.py
```

### Package Installation Issues
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Low Prediction Confidence
- Ensure the fingerprint image is clear and focused
- Use good lighting conditions
- Avoid blurry or low-quality images
- Try multiple images for comparison

## 🎯 Future Improvements

- [ ] Increase model accuracy through transfer learning
- [ ] Add more training data
- [ ] Implement ensemble methods
- [ ] Add image quality assessment
- [ ] Export prediction reports
- [ ] Multi-language support
- [ ] Mobile app version

## 📝 License

This project is for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 👨‍💻 Author

Created with ❤️ using Streamlit & TensorFlow

---

**Note**: This system achieves ~60% accuracy. For medical purposes, always consult healthcare professionals and use certified diagnostic methods.
"# Blood_group_detection_using_FingerPrint" 
