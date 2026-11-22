"""
Batch Document Classification Page
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import streamlit as st
import pandas as pd
import io
from document_extractor import DocumentExtractor
from predict import DocumentClassifier

# Page config
st.set_page_config(
    page_title="Batch Classification - Document Classifier",
    page_icon="📦",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .batch-summary {
        background: linear-gradient(135deg, #2E86C1 0%, #1B4F72 100%);
        color: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .file-item {
        background: white;
        border: 1px solid #E8E8E8;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    .file-item.success { border-left: 4px solid #27AE60; }
    .file-item.error { border-left: 4px solid #E74C3C; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_classifier():
    """Get cached classifier instance."""
    return DocumentClassifier()


def main():
    st.title("📦 Batch Document Classification")
    st.markdown("Upload multiple documents and classify them all at once.")

    classifier = get_classifier()

    # Check if model is loaded
    if not classifier._loaded:
        st.error("❌ Model not loaded. Please train the model first by running `python src/train.py`")
        st.stop()

    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['txt', 'pdf', 'docx'],
        accept_multiple_files=True,
        help="Supported formats: TXT, PDF, DOCX (max 10MB each)"
    )

    if uploaded_files:
        st.markdown(f"### 📁 {len(uploaded_files)} file(s) selected")

        # Show file list
        with st.expander("📋 Selected Files", expanded=True):
            for i, f in enumerate(uploaded_files):
                st.write(f"{i+1}. **{f.name}** ({f.size:,} bytes)")

        # Process button
        if st.button("🚀 Classify All Documents", type="primary", use_container_width=True):
            process_batch(classifier, uploaded_files)

    else:
        st.info("👆 Upload one or more documents to begin batch classification.")

        # Sample format info
        with st.expander("ℹ️ Supported Formats & Tips"):
            st.markdown("""
            **Supported Formats:**
            - **Text files (.txt)** - Plain text, any encoding
            - **PDF documents (.pdf)** - Text-based PDFs (scanned PDFs not supported)
            - **Word documents (.docx)** - Modern Word format

            **Tips:**
            - Maximum file size: 10 MB each
            - For best results, provide documents with at least 50 words
            - The system processes files in parallel for speed
            - Results can be exported as CSV
            """)


def process_batch(classifier, uploaded_files):
    """Process batch of uploaded files."""

    # Extract text from all files
    files_data = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.text("Extracting text from files...")
    for i, uploaded_file in enumerate(uploaded_files):
        progress_bar.progress((i + 1) / len(uploaded_files) * 0.3)
        file_bytes = uploaded_file.read()
        text, error = DocumentExtractor.extract_text(file_bytes, uploaded_file.name)
        files_data.append({
            'filename': uploaded_file.name,
            'text': text,
            'error': error,
            'success': error is None
        })

    # Count successes
    successful = [f for f in files_data if f['success']]
    failed = [f for f in files_data if not f['success']]

    if not successful:
        st.error("❌ No files could be processed successfully.")
        for f in failed:
            st.error(f"**{f['filename']}**: {f['error']}")
        return

    status_text.text(f"Classifying {len(successful)} document(s)...")

    # Prepare texts and IDs for classification
    texts = [f['text'] for f in successful]
    doc_ids = [f['filename'] for f in successful]

    # Classify batch
    progress_bar.progress(0.4)
    results = classifier.predict_batch(texts, doc_ids)
    progress_bar.progress(0.9)

    # Combine results with file info
    final_results = []
    for i, result in enumerate(results):
        file_info = successful[i]
        final_results.append({
            'filename': result['document_id'],
            'predicted_category': result['predicted_category'],
            'confidence': result['confidence'],
            'top_2_category': result['top_predictions'][1]['category'] if len(result['top_predictions']) > 1 else '',
            'top_2_prob': result['top_predictions'][1]['probability'] if len(result['top_predictions']) > 1 else 0,
            'top_3_category': result['top_predictions'][2]['category'] if len(result['top_predictions']) > 2 else '',
            'top_3_prob': result['top_predictions'][2]['probability'] if len(result['top_predictions']) > 2 else 0,
            'status': 'Success'
        })

    # Add failed files
    for f in failed:
        final_results.append({
            'filename': f['filename'],
            'predicted_category': 'ERROR',
            'confidence': 0,
            'top_2_category': '',
            'top_2_prob': 0,
            'top_3_category': '',
            'top_3_prob': 0,
            'status': f'Failed: {f["error"]}'
        })

    progress_bar.progress(1.0)
    status_text.text("Complete!")

    # Display results
    display_batch_results(final_results, successful, failed)


def display_batch_results(results, successful, failed):
    """Display batch classification results."""

    st.markdown("---")
    st.subheader("📊 Batch Classification Results")

    # Summary cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="batch-summary">
            <h3 style="margin: 0;">{len(results)}</h3>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Total Files</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="batch-summary" style="background: linear-gradient(135deg, #27AE60 0%, #1E8A49 100%);">
            <h3 style="margin: 0;">{len(successful)}</h3>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Successful</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="batch-summary" style="background: linear-gradient(135deg, #E74C3C 0%, #C0392B 100%);">
            <h3 style="margin: 0;">{len(failed)}</h3>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Failed</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        avg_conf = sum(r['confidence'] for r in results if r['status'] == 'Success') / max(len(successful), 1)
        st.markdown(f"""
        <div class="batch-summary" style="background: linear-gradient(135deg, #F39C12 0%, #D68910 100%);">
            <h3 style="margin: 0;">{avg_conf:.1%}</h3>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Avg Confidence</p>
        </div>
        """, unsafe_allow_html=True)

    # Results table
    st.markdown("### 📋 Detailed Results")

    df = pd.DataFrame(results)

    # Format for display
    display_df = df.copy()
    display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.2%}")
    display_df['top_2_prob'] = display_df['top_2_prob'].apply(lambda x: f"{x:.2%}" if x > 0 else '-')
    display_df['top_3_prob'] = display_df['top_3_prob'].apply(lambda x: f"{x:.2%}" if x > 0 else '-')

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "filename": st.column_config.TextColumn("Filename", width="medium"),
            "predicted_category": st.column_config.TextColumn("Predicted Category", width="medium"),
            "confidence": st.column_config.TextColumn("Confidence", width="small"),
            "top_2_category": st.column_config.TextColumn("2nd Choice", width="medium"),
            "top_2_prob": st.column_config.TextColumn("2nd Prob", width="small"),
            "top_3_category": st.column_config.TextColumn("3rd Choice", width="medium"),
            "top_3_prob": st.column_config.TextColumn("3rd Prob", width="small"),
            "status": st.column_config.TextColumn("Status", width="medium"),
        }
    )

    # Category distribution chart
    if successful:
        st.markdown("### 📈 Category Distribution")
        cat_counts = pd.DataFrame(results)
        cat_counts = cat_counts[cat_counts['status'] == 'Success']['predicted_category'].value_counts().reset_index()
        cat_counts.columns = ['Category', 'Count']

        import plotly.express as px
        fig = px.bar(
            cat_counts,
            x='Category',
            y='Count',
            title="Predicted Categories Distribution",
            color='Count',
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Export CSV
    st.markdown("### 💾 Export Results")

    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name=f"batch_classification_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

    # Show errors if any
    if failed:
        with st.expander(f"❌ Failed Files ({len(failed)})", expanded=True):
            for f in failed:
                st.error(f"**{f['filename']}**: {f['error']}")


if __name__ == "__main__":
    main()