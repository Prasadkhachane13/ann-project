"""
About Page
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import streamlit as st
import yaml
from pathlib import Path

# Page config
st.set_page_config(
    page_title="About - Document Classifier",
    page_icon="ℹ️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .about-section {
        background: white;
        border-radius: 10px;
        padding: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #E8E8E8;
        margin-bottom: 1.5rem;
    }
    .tech-badge {
        display: inline-block;
        background: #E8F0F8;
        color: #2E86C1;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0.2rem;
    }
    .limitation-item {
        background: #FEF5E7;
        border-left: 4px solid #F39C12;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    .limitation-item h4 {
        margin: 0 0 0.5rem 0;
        color: #B7950B;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_config():
    with open("config/config.yaml", 'r') as f:
        return yaml.safe_load(f)


def main():
    st.title("ℹ️ About")
    st.markdown("Project information, methodology, and technical details.")

    config = load_config()

    # Project Overview
    st.markdown("""
    <div class="about-section">
        <h2>📋 Project Overview</h2>
        <p><strong>Intelligent Document Classification System</strong> is an academic project that demonstrates
        the design and development of an Artificial Neural Network (ANN) based document classification system.</p>

        <h3>Objective</h3>
        <p>Build a complete end-to-end system that accepts textual documents, processes their contents,
        extracts meaningful numerical features using TF-IDF vectorization, and classifies them into
        one of 20 predefined categories using a Multilayer Perceptron (MLP) Artificial Neural Network.</p>

        <h3>Problem Statement</h3>
        <p>Automatically categorize documents from the 20 Newsgroups dataset into their respective newsgroups
        using deep learning techniques, achieving high accuracy while maintaining computational efficiency.</p>
    </div>
    """, unsafe_allow_html=True)

    # Technology Stack
    st.markdown("""
    <div class="about-section">
        <h2>🛠️ Technology Stack</h2>

        <h3>Core ML/DL</h3>
        <div>
            <span class="tech-badge">TensorFlow/Keras 2.13+</span>
            <span class="tech-badge">Scikit-learn 1.3+</span>
            <span class="tech-badge">NumPy 1.24+</span>
            <span class="tech-badge">Pandas 2.0+</span>
        </div>

        <h3>Visualization</h3>
        <div>
            <span class="tech-badge">Matplotlib 3.7+</span>
            <span class="tech-badge">Seaborn 0.12+</span>
            <span class="tech-badge">Plotly 5.17+</span>
        </div>

        <h3>Web Application</h3>
        <div>
            <span class="tech-badge">Streamlit 1.28+</span>
        </div>

        <h3>Document Processing</h3>
        <div>
            <span class="tech-badge">PyMuPDF (fitz)</span>
            <span class="tech-badge">python-docx</span>
        </div>

        <h3>Utilities</h3>
        <div>
            <span class="tech-badge">Joblib</span>
            <span class="tech-badge">PyYAML</span>
            <span class="tech-badge">NLTK</span>
            <span class="tech-badge">TQDM</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Dataset
    st.markdown("""
    <div class="about-section">
        <h2>📊 Dataset: 20 Newsgroups</h2>

        <h3>Source</h3>
        <p>Official Scikit-learn dataset: <code>sklearn.datasets.fetch_20newsgroups</code></p>

        <h3>Statistics</h3>
        <ul>
            <li><strong>Total Documents:</strong> ~18,846</li>
            <li><strong>Categories:</strong> 20</li>
            <li><strong>Train/Test Split:</strong> Original split preserved (~11,314 train / ~7,532 test)</li>
            <li><strong>Validation Split:</strong> 20% of training data (stratified)</li>
        </ul>

        <h3>Categories</h3>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem;">
            <div>• alt.atheism</div>
            <div>• comp.graphics</div>
            <div>• comp.os.ms-windows.misc</div>
            <div>• comp.sys.ibm.pc.hardware</div>
            <div>• comp.sys.mac.hardware</div>
            <div>• comp.windows.x</div>
            <div>• misc.forsale</div>
            <div>• rec.autos</div>
            <div>• rec.motorcycles</div>
            <div>• rec.sport.baseball</div>
            <div>• rec.sport.hockey</div>
            <div>• sci.crypt</div>
            <div>• sci.electronics</div>
            <div>• sci.med</div>
            <div>• sci.space</div>
            <div>• soc.religion.christian</div>
            <div>• talk.politics.guns</div>
            <div>• talk.politics.mideast</div>
            <div>• talk.politics.misc</div>
            <div>• talk.religion.misc</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Methodology
    st.markdown("""
    <div class="about-section">
        <h2>🔬 Methodology</h2>

        <h3>1. Text Preprocessing</h3>
        <ul>
            <li>Lowercasing and Unicode normalization</li>
            <li>Removal of email addresses and URLs</li>
            <li>Newsgroup header removal (From, Subject, Organization, etc.)</li>
            <li>Quoted text removal (lines starting with > or |)</li>
            <li>Whitespace normalization</li>
        </ul>

        <h3>2. Feature Extraction (TF-IDF)</h3>
        <ul>
            <li>Max features: 15,000</li>
            <li>N-gram range: Unigrams + Bigrams (1, 2)</li>
            <li>Min document frequency: 2</li>
            <li>Max document frequency: 0.95</li>
            <li>Sublinear TF scaling: Enabled</li>
            <li>Stop words: English</li>
            <li>Token pattern: Words with 2+ letters</li>
        </ul>

        <h3>3. ANN Architecture (MLP)</h3>
        <pre style="background: #F8F9FA; padding: 1rem; border-radius: 8px; overflow-x: auto;">
Input Layer:     15,000 features (TF-IDF)
    ↓
Hidden Layer 1:  256 neurons, ReLU activation
    ↓
Dropout:         0.40
    ↓
Hidden Layer 2:  128 neurons, ReLU activation
    ↓
Dropout:         0.30
    ↓
Output Layer:    20 neurons, Softmax activation
        </pre>

        <h3>4. Training Configuration</h3>
        <ul>
            <li>Optimizer: Adam (learning rate = 0.001)</li>
            <li>Loss: Sparse Categorical Crossentropy</li>
            <li>Batch Size: 32</li>
            <li>Max Epochs: 40</li>
            <li>Early Stopping: Patience 5 (monitor val_loss)</li>
            <li>Learning Rate Scheduler: ReduceLROnPlateau (factor 0.5, patience 3)</li>
            <li>Random Seed: 42 (for reproducibility)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Model Artifacts
    st.markdown("""
    <div class="about-section">
        <h2>💾 Model Artifacts</h2>
        <p>The following artifacts are saved after training:</p>

        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
            <div style="background: #F8F9FA; padding: 1rem; border-radius: 8px;">
                <h4>document_classifier.keras</h4>
                <p>Trained Keras model with architecture and weights</p>
            </div>
            <div style="background: #F8F9FA; padding: 1rem; border-radius: 8px;">
                <h4>tfidf_vectorizer.joblib</h4>
                <p>Fitted TF-IDF vectorizer for feature extraction</p>
            </div>
            <div style="background: #F8F9FA; padding: 1rem; border-radius: 8px;">
                <h4>label_encoder.joblib</h4>
                <p>Label encoder for category name ↔ integer mapping</p>
            </div>
            <div style="background: #F8F9FA; padding: 1rem; border-radius: 8px;">
                <h4>training_history.json</h4>
                <p>Training metrics per epoch</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Limitations
    st.markdown("""
    <div class="about-section">
        <h2>⚠️ Limitations</h2>

        <div class="limitation-item">
            <h4>Header Information Leakage</h4>
            <p>Despite header removal, some metadata may remain in the text that could leak category information.
            The dataset's original headers contain strong signals (e.g., newsgroup names in headers).</p>
        </div>

        <div class="limitation-item">
            <h4>Vocabulary Size Constraint</h4>
            <p>TF-IDF is limited to 15,000 features. Rare but discriminative terms may be excluded,
            potentially affecting classification of specialized topics.</p>
        </div>

        <div class="limitation-item">
            <h4>Memory Constraints</h4>
            <p>Dense matrix conversion for training requires significant RAM (~2-3 GB for full dataset).
            On systems with <8 GB RAM, training may be slow or fail.</p>
        </div>

        <div class="limitation-item">
            <h4>No OCR Support</h4>
            <p>Scanned PDFs or image-based documents cannot be processed. Only text-extractable PDFs are supported.</p>
        </div>

        <div class="limitation-item">
            <h4>English Only</h4>
            <p>The system is designed for English text only. Non-English documents will not be classified accurately.</p>
        </div>

        <div class="limitation-item">
            <h4>Static Model</h4>
            <p>The model cannot learn from new data without retraining. Online/incremental learning is not supported.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Future Improvements
    st.markdown("""
    <div class="about-section">
        <h2>🚀 Future Improvements</h2>
        <ul>
            <li><strong>Transformer-based models:</strong> Replace MLP with BERT or similar for better contextual understanding</li>
            <li><strong>Incremental learning:</strong> Support online model updates with new data</li>
            <li><strong>Multi-language support:</strong> Extend to non-English documents</li>
            <li><strong>OCR integration:</strong> Add Tesseract OCR for scanned documents</li>
            <li><strong>Active learning:</strong> Implement uncertainty sampling for human-in-the-loop labeling</li>
            <li><strong>Model compression:</strong> Quantization and distillation for edge deployment</li>
            <li><strong>Explainability:</strong> Add SHAP/LIME explanations for predictions</li>
            <li><strong>Ensemble methods:</strong> Combine multiple models for improved robustness</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # References
    st.markdown("""
    <div class="about-section">
        <h2>📚 References</h2>
        <ol>
            <li>Scikit-learn 20 Newsgroups Dataset: <a href="https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset" target="_blank">https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset</a></li>
            <li>TF-IDF Vectorization: <a href="https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html" target="_blank">Scikit-learn Documentation</a></li>
            <li>TensorFlow/Keras: <a href="https://www.tensorflow.org/" target="_blank">https://www.tensorflow.org/</a></li>
            <li>Streamlit: <a href="https://streamlit.io/" target="_blank">https://streamlit.io/</a></li>
            <li>PyMuPDF: <a href="https://pymupdf.readthedocs.io/" target="_blank">https://pymupdf.readthedocs.io/</a></li>
            <li>python-docx: <a href="https://python-docx.readthedocs.io/" target="_blank">https://python-docx.readthedocs.io/</a></li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #6B7B8D; font-size: 0.9rem;">
        <p><strong>Intelligent Document Classification System</strong></p>
        <p>Academic Project — B.Tech Artificial Intelligence & Data Science</p>
        <p>Built with ❤️ using TensorFlow, Scikit-learn, and Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()