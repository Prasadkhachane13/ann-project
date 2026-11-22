"""
Prediction History Page
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import streamlit as st
import pandas as pd
import plotly.express as px
from predict import DocumentClassifier

# Page config
st.set_page_config(
    page_title="Prediction History - Document Classifier",
    page_icon="🕐",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .history-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #E8E8E8;
        margin-bottom: 0.5rem;
    }
    .history-header {
        font-weight: 600;
        color: #1B2A4A;
        margin-bottom: 0.5rem;
    }
    .history-meta {
        font-size: 0.85rem;
        color: #6B7B8D;
    }
    .empty-state {
        text-align: center;
        padding: 3rem;
        color: #6B7B8D;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_classifier():
    """Get cached classifier instance."""
    return DocumentClassifier()


def main():
    st.title("🕐 Prediction History")
    st.markdown("View and manage your document classification history.")

    classifier = get_classifier()

    # Load history
    history = classifier.get_history()

    if not history:
        st.markdown("""
        <div class="empty-state">
            <h3>📭 No Prediction History</h3>
            <p>Your prediction history will appear here after you classify documents.</p>
            <p>Go to <strong>Single Classification</strong> or <strong>Batch Classification</strong> to start classifying.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Convert to DataFrame
    df = pd.DataFrame(history)

    # Parse timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['time'] = df['timestamp'].dt.time

    # Summary stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Predictions", len(df))

    with col2:
        unique_docs = df['document_id'].nunique()
        st.metric("Unique Documents", unique_docs)

    with col3:
        avg_conf = df['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_conf:.1%}")

    with col4:
        top_cat = df['predicted_category'].mode()[0] if len(df) > 0 else "N/A"
        st.metric("Most Predicted", top_cat)

    # Filters
    st.markdown("### 🔍 Filters")

    col1, col2, col3 = st.columns(3)

    with col1:
        date_range = st.date_input(
            "Date Range",
            value=(df['date'].min(), df['date'].max()),
            min_value=df['date'].min(),
            max_value=df['date'].max()
        )

    with col2:
        categories = ['All'] + sorted(df['predicted_category'].unique().tolist())
        selected_cat = st.selectbox("Category", categories)

    with col3:
        min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.05)

    # Apply filters
    filtered_df = df.copy()
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['date'] >= date_range[0]) &
            (filtered_df['date'] <= date_range[1])
        ]
    if selected_cat != 'All':
        filtered_df = filtered_df[filtered_df['predicted_category'] == selected_cat]
    filtered_df = filtered_df[filtered_df['confidence'] >= min_conf]

    # Sort by timestamp descending
    filtered_df = filtered_df.sort_values('timestamp', ascending=False)

    # Visualizations
    if len(filtered_df) > 0:
        st.markdown("### 📊 History Analytics")

        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            # Predictions over time
            daily_counts = filtered_df.groupby('date').size().reset_index(name='count')
            fig = px.line(
                daily_counts, x='date', y='count',
                title="Predictions Over Time",
                markers=True
            )
            fig.update_layout(height=300, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

        with viz_col2:
            # Category distribution
            cat_counts = filtered_df['predicted_category'].value_counts().reset_index()
            cat_counts.columns = ['Category', 'Count']
            fig = px.pie(
                cat_counts, values='Count', names='Category',
                title="Predicted Categories Distribution"
            )
            fig.update_layout(height=300, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

    # History table
    st.markdown(f"### 📋 History ({len(filtered_df)} entries)")

    if len(filtered_df) == 0:
        st.info("No entries match the current filters.")
    else:
        # Display options
        display_cols = ['timestamp', 'document_id', 'predicted_category', 'confidence']
        display_df = filtered_df[display_cols].copy()
        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.2%}")
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        display_df.columns = ['Timestamp', 'Document ID', 'Predicted Category', 'Confidence']

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Timestamp": st.column_config.TextColumn("Timestamp", width="medium"),
                "Document ID": st.column_config.TextColumn("Document ID", width="medium"),
                "Predicted Category": st.column_config.TextColumn("Category", width="medium"),
                "Confidence": st.column_config.TextColumn("Confidence", width="small"),
            }
        )

        # Export
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered History as CSV",
            data=csv,
            file_name=f"prediction_history_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Clear history
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🗑️ Clear All History", type="secondary", use_container_width=True):
            if st.session_state.get('confirm_clear', False):
                classifier.clear_history()
                st.success("History cleared!")
                st.rerun()
            else:
                st.session_state['confirm_clear'] = True
                st.warning("Click again to confirm")

    with col1:
        if st.session_state.get('confirm_clear', False):
            st.warning("⚠️ This will permanently delete all prediction history. Click again to confirm.")


if __name__ == "__main__":
    main()