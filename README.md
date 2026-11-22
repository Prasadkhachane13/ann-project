# Intelligent Document Classification System

**Design and Development of an Artificial Neural Network-Based Intelligent Document Classification System**

An end-to-end academic project demonstrating document classification using a Multilayer Perceptron (MLP) Artificial Neural Network with TF-IDF feature extraction on the 20 Newsgroups dataset.

---

## 📋 Problem Statement

Automatically classify textual documents into one of 20 predefined categories using deep learning techniques. The system should accept raw text documents, preprocess them, extract numerical features, and predict the correct category with high accuracy.

## 🎯 Objectives

1. **Data Acquisition**: Download and process the 20 Newsgroups dataset (~18,846 documents, 20 categories)
2. **Exploratory Data Analysis**: Comprehensive analysis of dataset characteristics, distributions, and quality
3. **Text Preprocessing**: Clean text, remove headers/footers/quotes, apply TF-IDF vectorization
4. **ANN Development**: Build and train a Multilayer Perceptron (MLP) using TensorFlow/Keras
5. **Model Evaluation**: Thorough evaluation with accuracy, precision, recall, F1-score, confusion matrix
6. **Web Application**: Deploy as a Streamlit web application for interactive classification
7. **Documentation**: Complete academic project documentation

---

## 📊 Dataset

**Source**: [20 Newsgroups Dataset](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset) via Scikit-learn

| Statistic | Value |
|-----------|-------|
| Total Documents | ~18,846 |
| Categories | 20 |
| Train/Test Split | Original split preserved (~11,314 / ~7,532) |
| Validation Split | 20% of training (stratified) |

### Categories

1. `alt.atheism`
2. `comp.graphics`
3. `comp.os.ms-windows.misc`
4. `comp.sys.ibm.pc.hardware`
5. `comp.sys.mac.hardware`
6. `comp.windows.x`
7. `misc.forsale`
8. `rec.autos`
9. `rec.motorcycles`
10. `rec.sport.baseball`
11. `rec.sport.hockey`
12. `sci.crypt`
13. `sci.electronics`
14. `sci.med`
15. `sci.space`
16. `soc.religion.christian`
17. `talk.politics.guns`
18. `talk.politics.mideast`
19. `talk.politics.misc`
20. `talk.religion.misc`

---

## 🏗️ System Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│   Raw Text  │────▶│ Preprocessing │────▶│  TF-IDF     │────▶│   MLP ANN    │
│  Document   │     │  (Cleaning)   │     │  Vectorizer │     │  Classifier  │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
                                                                        │
                                                                        ▼
                                                               ┌──────────────┐
                                                               │  Prediction  │
                                                               │  (Category)  │
                                                               └──────────────┘
```

### Components

1. **Text Preprocessing**
   - Lowercasing & Unicode normalization
   - Email/URL removal
   - Newsgroup header removal (From, Subject, Organization, etc.)
   - Quoted text removal (> and | prefixed lines)
   - Whitespace normalization

2. **Feature Extraction (TF-IDF)**
   - Max features: 15,000
   - N-gram range: (1, 2) - unigrams + bigrams
   - Min DF: 2, Max DF: 0.95
   - Sublinear TF scaling
   - English stop words
   - Fitted ONLY on training data

3. **ANN Architecture (MLP)**
   ```
   Input:     15,000 features
       ↓
   Dense(256, ReLU) + Dropout(0.40)
       ↓
   Dense(128, ReLU) + Dropout(0.30)
       ↓
   Dense(20, Softmax)
   ```

4. **Training Configuration**
   - Optimizer: Adam (lr=0.001)
   - Loss: Sparse Categorical Crossentropy
   - Batch Size: 32
   - Max Epochs: 40
   - Early Stopping: Patience 5 (val_loss)
   - LR Scheduler: ReduceLROnPlateau (factor 0.5, patience 3)

---

## 🛠️ Technology Stack

| Category | Technologies |
|----------|--------------|
| **Deep Learning** | TensorFlow/Keras 2.13+ |
| **ML Utilities** | Scikit-learn 1.3+, NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Web App** | Streamlit 1.28+ |
| **Document Processing** | PyMuPDF, python-docx |
| **Serialization** | Joblib, JSON, YAML |
| **Testing** | Pytest |

---

## 📁 Project Structure

```
Document_Classification_ANN/
│
├── app/                          # Streamlit Web Application
│   ├── Home.py                   # Main entry point
│   ├── pages/                    # Application pages
│   │   ├── 1_Single_Classification.py
│   │   ├── 2_Batch_Classification.py
│   │   ├── 3_Category_Explorer.py
│   │   ├── 4_Model_Performance.py
│   │   ├── 5_Prediction_History.py
│   │   └── 6_About.py
│   └── components/               # Reusable UI components
│
├── data/
│   ├── raw/                      # Raw dataset cache
│   └── processed/                # Processed data (TF-IDF matrices, labels)
│
├── notebooks/                    # Jupyter Notebooks
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_text_preprocessing.ipynb
│   ├── 03_ann_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── models/                       # Trained model artifacts
│   ├── document_classifier.keras
│   ├── tfidf_vectorizer.joblib
│   └── label_encoder.joblib
│
├── src/                          # Source code modules
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── ann_model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── document_extractor.py
│
├── results/
│   ├── figures/                  # Visualization outputs
│   ├── metrics/                  # Evaluation metrics (JSON)
│   └── reports/                  # Detailed reports (CSV/TXT)
│
├── tests/                        # Unit tests
│   ├── test_preprocessing.py
│   ├── test_model_loading.py
│   └── test_predictions.py
│
├── config/
│   └── config.yaml               # Configuration file
│
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── PROJECT_REPORT.md             # Academic project report
├── .gitignore
└── run_app.bat                   # Windows launcher
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- Windows 11 / Linux / macOS
- 8 GB RAM minimum (16 GB recommended for training)
- ~2 GB disk space

### Setup Steps

```bash
# 1. Clone/extract project
cd Document_Classification_ANN

# 2. Create virtual environment
python -m venv venv

# 3. Activate environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download NLTK data (for stopwords)
python -c "import nltk; nltk.download('stopwords')"
```

---

## 🚀 Usage

### 1. Train the Model

```bash
# Run complete training pipeline
python src/train.py

# This will:
# - Download 20 Newsgroups dataset
# - Preprocess text (clean, TF-IDF)
# - Train MLP ANN
# - Save model and artifacts
# - Generate training visualizations
```

### 2. Evaluate the Model

```bash
# Run evaluation on test set
python src/evaluate.py

# This will:
# - Load trained model
# - Predict on test set
# - Compute comprehensive metrics
# - Generate confusion matrix, per-class metrics
# - Save all results and visualizations
```

### 3. Launch Web Application

```bash
# Start Streamlit app
streamlit run app/Home.py

# Or use Windows batch file
run_app.bat
```

The app will open at `http://localhost:8501`

### 4. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_preprocessing.py -v
```

---

## 🌐 Web Application Features

| Page | Description |
|------|-------------|
| **Home** | Dashboard with system overview, metrics, navigation |
| **Single Classification** | Classify individual documents (text input or file upload) |
| **Batch Classification** | Upload multiple files, classify all, export CSV |
| **Category Explorer** | Browse all 20 categories with descriptions & statistics |
| **Model Performance** | View metrics, confusion matrix, training curves |
| **Prediction History** | Review past predictions with timestamps |
| **About** | Project details, methodology, limitations, references |

### Supported File Formats
- **Text** (.txt)
- **PDF** (.pdf) - text-based only (no OCR)
- **Word** (.docx)

---

## 📈 Results

### Expected Performance (approximate)

| Metric | Value |
|--------|-------|
| Test Accuracy | ~85-90% |
| Macro F1-Score | ~83-88% |
| Weighted F1-Score | ~85-90% |
| Top-5 Accuracy | ~95%+ |

*Note: Actual results may vary based on hardware, random seed, and training duration.*

### Generated Artifacts

After training and evaluation:

```
models/
├── document_classifier.keras      # Trained model
├── tfidf_vectorizer.joblib        # TF-IDF vectorizer
├── label_encoder.joblib           # Label encoder
└── model_config.yaml              # Model configuration

results/
├── figures/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── per_class_metrics.png
│   ├── category_distribution.png
│   └── ...
├── metrics/
│   ├── training_history.json
│   └── evaluation_metrics.json
└── reports/
    ├── classification_report.csv
    ├── classification_report.txt
    ├── confusion_matrix.csv
    └── ...
```

---

## ⚠️ Limitations

1. **Header Leakage**: Despite header removal, some metadata may remain
2. **Vocabulary Limit**: TF-IDF limited to 15,000 features
3. **Memory Usage**: Dense matrix conversion requires ~2-3 GB RAM
4. **No OCR**: Scanned/image-based PDFs not supported
5. **English Only**: Designed for English text only
6. **Static Model**: No online/incremental learning support

---

## 🔮 Future Improvements

- Transformer-based models (BERT, DistilBERT)
- Incremental/online learning
- Multi-language support
- OCR integration (Tesseract)
- Model compression (quantization, distillation)
- Explainability (SHAP/LIME)
- Ensemble methods
- Active learning

---

## 📚 References

1. [Scikit-learn 20 Newsgroups Dataset](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset)
2. [TF-IDF Vectorization - Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
3. [TensorFlow/Keras Documentation](https://www.tensorflow.org/)
4. [Streamlit Documentation](https://streamlit.io/)
5. [PyMuPDF Documentation](https://pymupdf.readthedocs.io/)
6. [python-docx Documentation](https://python-docx.readthedocs.io/)

---

## 👨‍💻 Author

Academic Project - B.Tech Artificial Intelligence & Data Science

---

## 📄 License

This project is for academic/educational purposes.