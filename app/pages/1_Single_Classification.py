"""
Single Document Classification Page
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from document_extractor import DocumentExtractor
from predict import DocumentClassifier

# Page config
st.set_page_config(
    page_title="Single Classification - Document Classifier",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .result-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #E8E8E8;
        margin-bottom: 1rem;
    }
    .prediction-main {
        background: linear-gradient(135deg, #2E86C1 0%, #1B4F72 100%);
        color: white;
        border-radius: 10px;
        padding: 2rem;
        margin-bottom: 1.5rem;
    }
    .confidence-bar {
        height: 8px;
        border-radius: 4px;
        background: linear-gradient(90deg, #2E86C1 0%, #27AE60 100%);
    }
    .stTextArea textarea {
        font-family: 'Monospace', monospace;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_classifier():
    """Get cached classifier instance."""
    return DocumentClassifier()


def main():
    st.title("🔍 Single Document Classification")
    st.markdown("Classify individual documents by entering text or uploading files.")

    classifier = get_classifier()

    # Check if model is loaded
    if not classifier._loaded:
        st.error("❌ Model not loaded. Please train the model first by running `python src/train.py`")
        st.stop()

    # Input method selection
    input_method = st.radio(
        "Input Method",
        ["✍️ Text Input", "📁 File Upload"],
        horizontal=True
    )

    document_text = ""
    doc_id = "manual_input"

    if input_method == "✍️ Text Input":
        document_text = st.text_area(
            "Enter document text:",
            height=200,
            placeholder="Paste your document text here...\n\nExample: The space shuttle launch was delayed due to technical issues with the main engines. NASA engineers are working to resolve the problem..."
        )
        doc_id = st.text_input("Document ID (optional):", value="manual_input")

    else:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['txt', 'pdf', 'docx'],
            help="Supported formats: TXT, PDF, DOCX (max 10MB)"
        )

        if uploaded_file is not None:
            doc_id = uploaded_file.name
            file_bytes = uploaded_file.read()

            # Extract text
            with st.spinner("Extracting text from file..."):
                text, error = DocumentExtractor.extract_text(file_bytes, uploaded_file.name)

            if error:
                st.error(f"❌ {error}")
                if "no extractable text" in error.lower():
                    st.info("💡 This PDF appears to be scanned or image-based. OCR is not supported in this version.")
                document_text = ""
            else:
                document_text = text
                st.success(f"✅ Text extracted successfully ({len(text):,} characters)")

                # Show preview
                with st.expander("📄 Extracted Text Preview"):
                    st.text(text[:2000] + ("..." if len(text) > 2000 else ""))

    # Classify button
    if st.button("🔍 Classify Document", type="primary", use_container_width=True, disabled=not document_text.strip()):
        if not document_text.strip():
            st.warning("Please enter or upload a document first.")
        else:
            with st.spinner("Classifying document..."):
                result = classifier.predict_single(document_text, doc_id)

            # Display results
            st.markdown("---")
            st.subheader("📊 Classification Results")

            # Main prediction card
            st.markdown(f"""
            <div class="prediction-main">
                <h2 style="margin: 0 0 0.5rem 0;">{result['predicted_category']}</h2>
                <p style="margin: 0; opacity: 0.9; font-size: 1.1rem;">Confidence: {result['confidence']:.2%}</p>
            </div>
            """, unsafe_allow_html=True)

            # Confidence bar
            st.markdown(f"""
            <div class="result-card">
                <h4>Confidence Level</h4>
                <div style="background: #E8E8E8; border-radius: 4px; height: 12px; overflow: hidden;">
                    <div class="confidence-bar" style="width: {result['confidence']*100}%;"></div>
                </div>
                <p style="margin-top: 0.5rem; color: #6B7B8D;">{result['confidence']:.2%} confident in this prediction</p>
            </div>
            """, unsafe_allow_html=True)

            # Top predictions table
            st.markdown("### Top 5 Predictions")

            pred_df = pd.DataFrame(result['top_predictions'])
            pred_df['probability'] = pred_df['probability'].apply(lambda x: f"{x:.4f} ({x:.2%})")
            pred_df.columns = ['Rank', 'Category', 'Probability']

            st.dataframe(
                pred_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Rank": st.column_config.NumberColumn("Rank", width="small"),
                    "Category": st.column_config.TextColumn("Category", width="medium"),
                    "Probability": st.column_config.TextColumn("Probability", width="medium"),
                }
            )

            # Visualization
            fig = px.bar(
                result['top_predictions'],
                x='probability',
                y='category',
                orientation='h',
                title="Prediction Probabilities",
                labels={'probability': 'Probability', 'category': 'Category'},
                color='probability',
                color_continuous_scale='Blues'
            )
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=350,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            fig.update_traces(texttemplate='%{x:.2%}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

            # Document info
            with st.expander("📝 Document Information"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Document ID:** {result['document_id']}")
                    st.write(f"**Text Length:** {len(document_text):,} characters")
                    st.write(f"**Words:** {len(document_text.split()):,}")
                with col2:
                    st.write(f"**Cleaned Text Preview:**")
                    st.code(result['cleaned_text_preview'], language=None)

    # Sidebar info
    with st.sidebar:
        st.markdown("### ℹ️ Information")
        st.info("""
        **Supported Formats:**
        - Text files (.txt)
        - PDF documents (.pdf)
        - Word documents (.docx)

        **Tips for best results:**
        - Provide sufficient text (at least 50 words)
        - Avoid heavily formatted documents
        - Scanned PDFs require OCR (not supported)
        """)

        st.markdown("### 📂 Categories")
        categories = classifier.get_categories()
        for cat in sorted(categories):
            st.caption(f"• {cat}")


if __name__ == "__main__":
    main()