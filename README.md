# Multi-Label Retinal Disease Classification System

A deep learning pipeline for multi-label retinal disease classification that combines computer vision with biomedical natural language processing using BioBERT embeddings and transformer architecture.

## 🎯 Overview

This project implements a novel approach to medical image classification by integrating:
- **Visual Feature Extraction**: DenseNet201 with multi-scale feature processing
- **Biomedical Knowledge**: BioBERT embeddings for disease semantic understanding
- **Transformer Architecture**: Multi-head attention for feature fusion
- **Multi-label Classification**: Simultaneous detection of multiple retinal conditions

## 🏥 Medical Conditions Detected

The model can classify 20 different retinal conditions:

| Abbreviation | Full Name |
|-------------|-----------|
| DR | Diabetic Retinopathy |
| NORMAL | Normal |
| MH | Media Haze |
| ODC | Optic Disc Coloboma |
| TSLN | Tessellation |
| ARMD | Age Related Macular Degeneration |
| DN | Drusen |
| MYA | Myopia |
| BRVO | Branch Retinal Vein Occlusion |
| ODP | Optic Disc Pallor |
| CRVO | Central Retinal Vein Occlusion |
| CNV | Choroidal Neovascularization |
| RS | Retinitis |
| ODE | Optic Disc Edema |
| LS | Laser Scars |
| CSR | Central Serous Retinopathy |
| HTR | Hypertensive Retinopathy |
| ASR | Artificial Silicon Retina |
| CRS | Chorioretinitis |
| OTHER | Other Conditions |

## 🏗️ Architecture

### 1. Visual Feature Extraction
- **Base Model**: DenseNet201 (pre-trained on ImageNet)
- **Multi-Scale Feature Module (MSFM)**: Combines high and low-level features
- **Channel Attention Module (CAM)**: Enhances relevant visual features
- **Input**: 320×320×3 retinal fundus images

### 2. BioBERT Disease Embeddings
- **Model**: `dmis-lab/biobert-base-cased-v1.1`
- **Purpose**: Generate semantic embeddings for disease names
- **Output**: 512-dimensional disease representations
- **Fallback**: BERT-base-uncased if BioBERT unavailable

### 3. Transformer Fusion
- **Multi-Head Attention**: 8 attention heads
- **Encoder Layers**: 2 transformer encoder blocks
- **Feature Dimension**: 512
- **Combines**: Visual features + Disease embeddings

### 4. Classification Head
- **Architecture**: Dense layers with dropout regularization
- **Layers**: 1024 → 512 → 256 → 20 (output)
- **Activation**: Sigmoid (multi-label classification)
- **Dropout**: 0.5, 0.3, 0.2 respectively

## Architecture Pipeline

```python
Input Image (320x320x3)
         ↓
   DenseNet201 (ImageNet pretrained)
         ↓
Multi-scale Feature Extraction (MSFM)
         ↓
Channel Attention Module (CAM)
         ↓
Visual Features (512-dim)
         ↓
    Concatenate ← BioBERT Disease Embeddings (512-dim)
         ↓
  Transformer Encoder Layers (2x)
         ↓
   Global Mean Pooling
         ↓
 Classification Head (1024→512→256→20)
         ↓
   Disease Predictions (20 classes)
```   
## 📋 Requirements

### Core Dependencies
```python
tensorflow>=2.8.0
transformers>=4.20.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn
Pillow
```

### Optional (for Google Colab)
```python
google-colab
```

## 📊 Data Structure

### Expected Data Format
```
/content/
├── drive/MyDrive/
│   ├── mured.zip                    # Dataset archive
│   ├── train_data_modified.csv      # Training metadata
│   └── test_data_modified.csv       # Test metadata
└── images/images/                   # Extracted image directory
```

### CSV Structure
Required columns:
- `ID_2`: Image filename
- Disease columns: Binary labels for each of the 20 conditions


## 🔧 Configuration Options

### Model Parameters
- **Image Size**: 320×320 pixels (configurable)
- **Batch Size**: 16 (default)
- **Learning Rate**: 0.0001
- **Embedding Dimension**: 512
- **Transformer Heads**: 8
- **Dropout Rates**: [0.5, 0.3, 0.2]

### Training Parameters
- **Epochs**: 30 (with early stopping)
- **Patience**: 7 epochs
- **Validation Split**: 20%
- **Data Augmentation**: Rotation, shifts, zoom, flip

## 📈 Model Performance

The model is evaluated using multiple metrics:
- **Accuracy**: Overall classification accuracy
- **Precision**: Positive prediction accuracy
- **Recall**: Sensitivity to positive cases
- **AUC**: Area under ROC curve
- **Loss**: Binary cross-entropy loss

## 🔍 Key Features

### 1. **Robust Error Handling**
- Automatic fallback for BioBERT loading
- Alternative model options if primary fails
- Data path validation

### 2. **Data Augmentation**
- Rotation (40°), shifts (20%), shear, zoom
- Horizontal flipping for training robustness
- Maintains aspect ratio

### 3. **Memory Efficient**
- Batch processing with generators
- Configurable batch sizes
- Optional data sampling

### 4. **Modular Design**
- Separate classes for different components
- Easy to modify individual modules
- Clear separation of concerns

## 🔬 Research Contributions

1. **Novel Architecture**: First integration of BioBERT with medical image classification
2. **Multi-Modal Learning**: Combines visual and textual medical knowledge
3. **Attention Mechanisms**: Transformer-based feature fusion
4. **Clinical Relevance**: Addresses real-world retinal disease detection

## 🆘 Troubleshooting

### Common Issues

**1. BioBERT Loading Error**
```python
# Solution: The script automatically falls back to BERT-base-uncased
# Ensure transformers library is properly installed
```

**2. CUDA Memory Error**
```python
# Reduce batch size in create_data_generators()
batch_size = 8  # Instead of 16
```

**3. Data Path Issues**
```python
# Ensure proper file structure and update paths in load_and_prepare_data()
```

**4. Transformer Compatibility**
```python
# Update transformers library
pip install --upgrade transformers
```

## 📞 Support

For questions and support:

- Email: abdulhadizeeshan79@gmail.com

**Note**: This model is designed for research purposes. For clinical applications, ensure proper validation and regulatory compliance.
