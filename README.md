# image_classification_object_model

## 📌 Overview
This project implements an image classification system using deep learning to identify objects in images. The system includes:
- A training pipeline using transfer learning with MobileNetV2
- A Flask web application for image uploads and predictions
- A standalone prediction script

## 📂 Project Structure

```
image_classifier/
│
├── dataset/                  # Training data
│   └── train/                # Main training folder
│       ├── cat/              # Cat images (jpg/png)
│       ├── dog/              # Dog images
│       └── bird/             # Bird images
│
├── models/                   # Saved models
│   └── efficientnet_model.h5 # Trained model
│
├── static/                   # Web assets
│   └── uploads/              # User-uploaded images
│
├── templates/                # HTML templates
│   └── index.html            # Web interface
│
├── train.py                  # Model training script
├── predict.py                # Standalone prediction script
├── app.py                    # Flask web application
└── requirements.txt          # Python dependencies
```

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/image-classifier.git
   cd image-classifier
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Dataset Preparation

1. Create the directory structure:
   ```bash
   mkdir -p dataset/train/{cat,dog,bird}
   ```

2. Add images to each class folder:
   - Minimum 50-100 images per class recommended
   - Supported formats: JPG, PNG, JPEG
   - Ideal size: 224x224 pixels

## 🚀 Usage

### 1. Training the Model
```bash
python train.py
```

Training process:
1. Loads images from `dataset/train/`
2. Uses 80% for training, 20% for validation
3. Saves model to `models/efficientnet_model.h5`

### 2. Web Application
```bash
python app.py
```
- Access at: http://localhost:5000
- Features:
  - Drag-and-drop image upload
  - Real-time predictions
  - Confidence visualization

### 3. Standalone Prediction
```bash
python predict.py path/to/your/image.jpg
```
Outputs:
- Predicted class
- Confidence percentage
- All class probabilities

## 🔧 Customization

### To change classes:
1. Modify folder structure in `dataset/train/`
2. Update `CLASS_NAMES` in:
   - `train.py` (line 10)
   - `app.py` (line 14)
   - `predict.py` (line 10)

### To use different model:
1. Replace `MobileNetV2` in `train.py` with:
   ```python
   from tensorflow.keras.applications import EfficientNetB0
   base_model = EfficientNetB0(weights='imagenet', include_top=False)
   ```

## 📈 Performance Tips

1. For better accuracy:
   - Increase training images (100+ per class)
   - Add data augmentation in `train.py`
   - Train for more epochs (50-100)

2. To reduce overfitting:
   - Add Dropout layers
   - Use image augmentation
   - Implement early stopping

## 🤖 Model Architecture

```
MobileNetV2 (pretrained)
└── GlobalAveragePooling2D
    └── Dense(1024, relu)
        └── Dense(3, softmax)  # Output layer
```
