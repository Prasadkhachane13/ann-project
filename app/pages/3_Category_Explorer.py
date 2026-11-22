"""
Category Explorer Page
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import streamlit as st
import pandas as pd
import plotly.express as px
import yaml
from pathlib import Path
from predict import DocumentClassifier

# Page config
st.set_page_config(
    page_title="Category Explorer - Document Classifier",
    page_icon="📂",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .category-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #E8E8E8;
        height: 100%;
        transition: all 0.2s;
    }
    .category-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.12);
        border-color: #2E86C1;
    }
    .category-name {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1B2A4A;
        margin-bottom: 0.5rem;
    }
    .category-group {
        font-size: 0.8rem;
        color: #2E86C1;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .category-desc {
        font-size: 0.9rem;
        color: #6B7B8D;
        line-height: 1.5;
    }
    .stat-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #E8E8E8;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_category_data():
    """Load category information and dataset statistics."""
    config_path = "config/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Category descriptions
    descriptions = {
        'comp.graphics': 'Computer graphics, 3D rendering, and visualization techniques',
        'comp.os.ms-windows.misc': 'Microsoft Windows operating system discussions',
        'comp.sys.ibm.pc.hardware': 'IBM PC compatible hardware topics',
        'comp.sys.mac.hardware': 'Apple Macintosh hardware discussions',
        'comp.windows.x': 'X Window System and Unix graphics',
        'rec.autos': 'Automobiles, cars, and automotive topics',
        'rec.motorcycles': 'Motorcycles and motorcycling',
        'rec.sport.baseball': 'Baseball sport discussions',
        'rec.sport.hockey': 'Ice hockey sport discussions',
        'sci.crypt': 'Cryptography and encryption',
        'sci.electronics': 'Electronics and electrical engineering',
        'sci.med': 'Medicine, health, and medical research',
        'sci.space': 'Space exploration, astronomy, and space science',
        'misc.forsale': 'For-sale advertisements and commerce',
        'talk.politics.guns': 'Gun control and firearms politics',
        'talk.politics.mideast': 'Middle East politics and conflicts',
        'talk.politics.misc': 'General political discussions',
        'talk.religion.misc': 'Religious discussions and theology',
        'alt.atheism': 'Atheism and secularism discussions',
        'soc.religion.christian': 'Christianity and Christian theology',
    }

    # Group categories
    groups = {
        'comp.*': ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
                   'comp.sys.mac.hardware', 'comp.windows.x'],
        'rec.*': ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'],
        'sci.*': ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'],
        'misc.*': ['misc.forsale'],
        'talk.*': ['talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc',
                   'talk.religion.misc'],
        'alt.*': ['alt.atheism'],
        'soc.*': ['soc.religion.christian'],
    }

    # Try to load dataset stats
    stats = {}
    try:
        from src.data_loader import DataLoader
        loader = DataLoader()
        dataset = loader.load_dataset()
        df = loader.create_dataframe(dataset)
        for cat in df['category'].unique():
            cat_data = df[df['category'] == cat]
            stats[cat] = {
                'total': len(cat_data),
                'train': len(cat_data[cat_data['dataset_split'] == 'train']),
                'test': len(cat_data[cat_data['dataset_split'] == 'test']),
            }
    except Exception:
        pass

    return descriptions, groups, stats


def main():
    st.title("📂 Category Explorer")
    st.markdown("Explore all 20 Newsgroups categories with descriptions and statistics.")

    descriptions, groups, stats = load_category_data()

    # Get classifier for model info
    classifier = DocumentClassifier()

    # Summary stats
    if stats:
        st.markdown("### 📊 Dataset Statistics")

        total_docs = sum(s['total'] for s in stats.values())
        total_train = sum(s['train'] for s in stats.values())
        total_test = sum(s['test'] for s in stats.values())

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div style="font-size: 2rem; font-weight: 700; color: #1B2A4A;">{total_docs:,}</div>
                <div style="color: #6B7B8D;">Total Documents</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div style="font-size: 2rem; font-weight: 700; color: #27AE60;">{total_train:,}</div>
                <div style="color: #6B7B8D;">Training</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="stat-card">
                <div style="font-size: 2rem; font-weight: 700; color: #E74C3C;">{total_test:,}</div>
                <div style="color: #6B7B8D;">Test</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="stat-card">
                <div style="font-size: 2rem; font-weight: 700; color: #2E86C1;">{len(stats)}</div>
                <div style="color: #6B7B8D;">Categories</div>
            </div>
            """, unsafe_allow_html=True)

        # Distribution chart
        st.markdown("### 📈 Category Distribution")
        dist_df = pd.DataFrame([
            {'Category': cat, 'Total': s['total'], 'Train': s['train'], 'Test': s['test']}
            for cat, s in stats.items()
        ]).sort_values('Total', ascending=True)

        fig = px.bar(
            dist_df,
            x='Total',
            y='Category',
            orientation='h',
            title="Documents per Category",
            color='Total',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=500, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    # Categories by group
    st.markdown("### 🗂️ Categories by Group")

    for group_name, categories in groups.items():
        with st.expander(f"**{group_name}** ({len(categories)} categories)", expanded=True):
            cols = st.columns(min(3, len(categories)))
            for i, cat in enumerate(categories):
                with cols[i % 3]:
                    desc = descriptions.get(cat, 'No description available')
                    stat = stats.get(cat, {})

                    card_html = f"""
                    <div class="category-card">
                        <div class="category-group">{group_name}</div>
                        <div class="category-name">{cat}</div>
                        <div class="category-desc">{desc}</div>
                    """

                    if stat:
                        card_html += f"""
                        <hr style="margin: 0.8rem 0; border-color: #E8E8E8;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #6B7B8D;">
                            <span>Total: {stat['total']:,}</span>
                            <span>Train: {stat['train']:,}</span>
                            <span>Test: {stat['test']:,}</span>
                        </div>
                        """

                    card_html += "</div>"
                    st.markdown(card_html, unsafe_allow_html=True)

    # Model categories
    if classifier._loaded:
        st.markdown("---")
        st.markdown("### 🤖 Model Categories")
        st.info(f"The trained model can classify documents into these **{len(classifier.get_categories())} categories**:")

        model_cats = sorted(classifier.get_categories())
        cols = st.columns(4)
        for i, cat in enumerate(model_cats):
            with cols[i % 4]:
                st.caption(f"• {cat}")


if __name__ == "__main__":
    main()