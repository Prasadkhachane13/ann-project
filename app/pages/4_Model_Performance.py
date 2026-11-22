"""
Model Performance Page
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import streamlit as st
import pandas as pd
import json
import yaml
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Model Performance - Document Classifier",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #E8E8E8;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1B2A4A;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6B7B8D;
        margin-top: 0.5rem;
    }
    .perf-section {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #E8E8E8;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_metrics():
    """Load evaluation metrics and training history."""
    metrics = {}
    history = {}

    # Load evaluation metrics
    metrics_path = Path("results/metrics/evaluation_metrics.json")
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

    # Load training history
    history_path = Path("results/metrics/training_history.json")
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)

    # Load config for category names
    config_path = "config/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return metrics, history, config


def main():
    st.title("📈 Model Performance")
    st.markdown("Comprehensive evaluation metrics and training visualization.")

    metrics, history, config = load_metrics()

    if not metrics:
        st.warning("⚠️ No evaluation metrics found. Please run `python src/evaluate.py` first.")
        st.stop()

    # Overall Metrics
    st.markdown("### 🎯 Overall Metrics")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    metrics_data = [
        ("Accuracy", metrics.get('accuracy', 0), "#2E86C1"),
        ("Top-5 Acc", metrics.get('top5_accuracy', 0), "#27AE60"),
        ("Macro Prec", metrics.get('precision_macro', 0), "#F39C12"),
        ("Weighted Prec", metrics.get('precision_weighted', 0), "#8E44AD"),
        ("Macro Rec", metrics.get('recall_macro', 0), "#E74C3C"),
        ("Macro F1", metrics.get('f1_macro', 0), "#16A085"),
    ]

    for i, (label, value, color) in enumerate(metrics_data):
        with [col1, col2, col3, col4, col5, col6][i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: {color};">{value:.2%}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    # Training History
    if history:
        st.mark('<div class="perf-section">', unsafe_allow_html=True)
        st.markdown("### 📊 Training History")

        epochs = list(range(1, len(history['accuracy']) + 1))

        fig = go.Figure()

        # Accuracy
        fig.add_trace(go.Scatter(
            x=epochs, y=history['accuracy'],
            mode='lines+markers', name='Training Accuracy',
            line=dict(color='#2E86C1', width=2), marker=dict(size=6)
        ))
        fig.add_trace(go.Scatter(
            x=epochs, y=history['val_accuracy'],
            mode='lines+markers', name='Validation Accuracy',
            line=dict(color='#E74C3C', width=2), marker=dict(size=6)
        ))

        fig.update_layout(
            title='Model Accuracy',
            xaxis_title='Epoch',
            yaxis_title='Accuracy',
            yaxis=dict(range=[0, 1.05]),
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Loss
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=epochs, y=history['loss'],
            mode='lines+markers', name='Training Loss',
            line=dict(color='#2E86C1', width=2), marker=dict(size=6)
        ))
        fig2.add_trace(go.Scatter(
            x=epochs, y=history['val_loss'],
            mode='lines+markers', name='Validation Loss',
            line=dict(color='#E74C3C', width=2), marker=dict(size=6)
        ))

        fig2.update_layout(
            title='Model Loss',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Per-Class Metrics
    if 'per_class' in metrics:
        st.markdown('<div class="perf-section">', unsafe_allow_html=True)
        st.markdown("### 📋 Per-Category Performance")

        per_class = metrics['per_class']
        categories = config['dataset']['categories'] if config['dataset']['categories'] != 'all' else [
            'alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
            'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos',
            'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt',
            'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian',
            'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc',
            'talk.religion.misc'
        ]

        # Create DataFrame
        class_data = []
        for cat in categories:
            if cat in per_class:
                class_data.append({
                    'Category': cat,
                    'Precision': per_class[cat]['precision'],
                    'Recall': per_class[cat]['recall'],
                    'F1-Score': per_class[cat]['f1-score'],
                    'Support': per_class[cat]['support']
                })

        df = pd.DataFrame(class_data)

        # Bar chart for F1 scores
        fig = px.bar(
            df.sort_values('F1-Score'),
            x='F1-Score',
            y='Category',
            orientation='h',
            title="Per-Category F1-Scores",
            color='F1-Score',
            color_continuous_scale='RdYlGn',
            range_color=[0, 1]
        )
        fig.update_layout(height=500, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        # Detailed table
        st.markdown("#### Detailed Metrics Table")
        display_df = df.copy()
        for col in ['Precision', 'Recall', 'F1-Score']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Category": st.column_config.TextColumn("Category", width="medium"),
                "Precision": st.column_config.TextColumn("Precision", width="small"),
                "Recall": st.column_config.TextColumn("Recall", width="small"),
                "F1-Score": st.column_config.TextColumn("F1-Score", width="small"),
                "Support": st.column_config.NumberColumn("Support", width="small"),
            }
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Confusion Matrix
    if 'confusion_matrix' in metrics:
        st.markdown('<div class="perf-section">', unsafe_allow_html=True)
        st.markdown("### 🔄 Confusion Matrix")

        cm = metrics['confusion_matrix']
        categories = [
            'alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
            'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos',
            'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt',
            'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian',
            'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc',
            'talk.religion.misc'
        ]

        # Normalized confusion matrix
        import numpy as np
        cm_array = np.array(cm)
        cm_norm = cm_array.astype('float') / cm_array.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

        fig = px.imshow(
            cm_norm,
            x=categories,
            y=categories,
            color_continuous_scale='Blues',
            title="Normalized Confusion Matrix",
            labels=dict(x="Predicted", y="Actual", color="Proportion"),
            aspect="auto"
        )
        fig.update_layout(
            height=600,
            xaxis_tickangle=-45,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        fig.update_xaxes(side="bottom")
        st.plotly_chart(fig, use_container_width=True)

        # Show raw counts on hover
        st.caption("Hover over cells to see exact values. Diagonal represents correct predictions.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Model Configuration
    st.markdown('<div class="perf-section">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Model Configuration")

    model_config = config.get('model', {})
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Architecture**")
        arch_data = [
            ["Input Dimension", model_config.get('input_dim', 'N/A')],
            ["Hidden Layer 1", f"{model_config.get('hidden_layers', [{}])[0].get('units', 'N/A')} units, {model_config.get('hidden_layers', [{}])[0].get('activation', 'N/A')}, Dropout {model_config.get('hidden_layers', [{}])[0].get('dropout', 'N/A')}"],
            ["Hidden Layer 2", f"{model_config.get('hidden_layers', [{}])[1].get('units', 'N/A')} units, {model_config.get('hidden_layers', [{}])[1].get('activation', 'N/A')}, Dropout {model_config.get('hidden_layers', [{}])[1].get('dropout', 'N/A')}"],
            ["Output Layer", f"{model_config.get('output_units', 'N/A')} units, {model_config.get('output_activation', 'N/A')}"],
        ]
        st.table(pd.DataFrame(arch_data, columns=["Component", "Configuration"]))

    with col2:
        st.markdown("**Training Parameters**")
        train_data = [
            ["Optimizer", model_config.get('optimizer', 'N/A')],
            ["Learning Rate", model_config.get('learning_rate', 'N/A')],
            ["Loss Function", model_config.get('loss', 'N/A')],
            ["Batch Size", model_config.get('batch_size', 'N/A')],
            ["Max Epochs", model_config.get('max_epochs', 'N/A')],
            ["Early Stopping", f"Patience: {model_config.get('early_stopping_patience', 'N/A')}"],
        ]
        st.table(pd.DataFrame(train_data, columns=["Parameter", "Value"]))

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()