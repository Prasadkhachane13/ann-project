"""
Home Page - Intelligent Document Classification System

Main entry point for the Streamlit application.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import streamlit as st
import yaml
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Intelligent Document Classification System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
@st.cache_data
def load_config():
    with open("config/config.yaml", 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main styling */
    .main {
        background-color: #FAFAFA;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1B2A4A 0%, #2C3E6B 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
    }

    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }

    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #E8E8E8;
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.12);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1B2A4A;
        line-height: 1;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #6B7B8D;
        margin-top: 0.5rem;
        font-weight: 500;
    }

    .metric-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }

    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1B2A4A;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E8E8E8;
    }

    /* Info boxes */
    .info-box {
        background: #F0F4F8;
        border-left: 4px solid #2E86C1;
        padding: 1rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }

    .info-box h4 {
        margin: 0 0 0.5rem 0;
        color: #1B2A4A;
    }

    .info-box p {
        margin: 0;
        color: #4A5A6A;
        font-size: 0.95rem;
    }

    /* Navigation cards */
    .nav-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #E8E8E8;
        height: 100%;
        transition: all 0.2s;
        text-decoration: none;
        color: inherit;
    }

    .nav-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(46, 134, 193, 0.15);
        border-color: #2E86C1;
    }

    .nav-card-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1B2A4A;
        margin-bottom: 0.5rem;
    }

    .nav-card-desc {
        font-size: 0.9rem;
        color: #6B7B8D;
        line-height: 1.5;
    }

    /* Footer */
    .footer {
        margin-top: 3rem;
        padding: 1.5rem;
        background: #F0F4F8;
        border-radius: 10px;
        text-align: center;
        color: #6B7B8D;
        font-size: 0.85rem;
    }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #2E86C1 0%, #1B4F72 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(46, 134, 193, 0.4);
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main home page."""

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📄 Intelligent Document Classification System</h1>
        <p>AI-Powered Text Classification using Artificial Neural Networks</p>
    </div>
    """, unsafe_allow_html=True)

    # Load model info
    try:
        from predict import DocumentClassifier
        classifier = DocumentClassifier()
        model_info = classifier.get_model_info()
        model_loaded = model_info['model_loaded']
    except Exception:
        model_loaded = False
        model_info = {}

    # Status indicator
    if model_loaded:
        st.success("✅ Model loaded successfully - System ready for classification")
    else:
        st.warning("⚠️ Model not loaded. Please train the model first using `python src/train.py`")

    # Key Metrics Row
    st.markdown('<div class="section-header">📊 System Overview</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🏷️</div>
            <div class="metric-value">{model_info.get('num_categories', 20)}</div>
            <div class="metric-label">Categories</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">📚</div>
            <div class="metric-value">{model_info.get('vocabulary_size', 'N/A'):,}</div>
            <div class="metric-label">Vocabulary Size</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🧠</div>
            <div class="metric-value">MLP</div>
            <div class="metric-label">Architecture</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">⚡</div>
            <div class="metric-value">TF-IDF</div>
            <div class="metric-label">Feature Extraction</div>
        </div>
        """, unsafe_allow_html=True)

    # Navigation Cards
    st.markdown('<div class="section-header">🧭 Navigation</div>', unsafe_allow_html=True)

    nav_cols = st.columns(3)

    nav_items = [
        ("🔍 Single Document Classification", "Classify individual documents by pasting text or uploading files (TXT, PDF, DOCX).", "pages/1_Single_Classification.py"),
        ("📦 Batch Document Classification", "Upload multiple documents and classify them all at once with CSV export.", "pages/2_Batch_Classification.py"),
        ("📂 Category Explorer", "Explore all 20 newsgroup categories with descriptions and statistics.", "pages/3_Category_Explorer.py"),
        ("📈 Model Performance", "View detailed evaluation metrics, confusion matrix, and training curves.", "pages/4_Model_Performance.py"),
        ("🕐 Prediction History", "Review past predictions with timestamps and confidence scores.", "pages/5_Prediction_History.py"),
        ("ℹ️ About", "Project details, methodology, limitations, and technical information.", "pages/6_About.py"),
    ]

    for i, (title, desc, page) in enumerate(nav_items):
        with nav_cols[i % 3]:
            st.markdown(f"""
            <div class="nav-card" onclick="window.location.href='{page}'">
                <div class="nav-card-title">{title}</div>
                <div class="nav-card-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Open {title.split(' ', 1)[1]}", key=f"nav_{i}", use_container_width=True):
                st.switch_page(page)

    # Quick Start Guide
    st.markdown('<div class="section-header">🚀 Quick Start</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <h4>Getting Started</h4>
        <p>1. <strong>Train the model:</strong> Run <code>python src/train.py</code> to download the 20 Newsgroups dataset, preprocess text, and train the ANN.</p>
        <p>2. <strong>Evaluate:</strong> Run <code>python src/evaluate.py</code> to generate performance metrics and visualizations.</p>
        <p>3. <strong>Launch app:</strong> Run <code>streamlit run app/Home.py</code> to start the web interface.</p>
        <p>4. <strong>Classify:</strong> Navigate to <em>Single Document Classification</em> or <em>Batch Classification</em> to classify documents.</p>
    </div>
    """, unsafe_allow_html=True)

    # Project Info
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>📋 Dataset</h4>
            <p><strong>20 Newsgroups</strong> - ~18,846 documents across 20 categories covering topics like computers, science, politics, religion, and recreation.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
            <h4>🔧 Preprocessing</h4>
            <p>Text cleaning, header/footer removal, TF-IDF vectorization (15,000 features, unigrams + bigrams), sublinear TF scaling.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>🧠 Model Architecture</h4>
            <p>Multilayer Perceptron: Input(15K) → Dense(256, ReLU) → Dropout(0.4) → Dense(128, ReLU) → Dropout(0.3) → Dense(20, Softmax)</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
            <h4>⚙️ Training</h4>
            <p>Adam optimizer, lr=0.001, sparse categorical crossentropy, batch size 32, early stopping (patience=5), max 40 epochs.</p>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        <strong>Intelligent Document Classification System</strong> |
        Built with TensorFlow/Keras, Scikit-learn, Streamlit |
        Academic Project - B.Tech AI & Data Science
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()