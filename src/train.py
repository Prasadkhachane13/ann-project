"""
Training Script for Document Classification ANN

This script orchestrates the complete training pipeline:
1. Load and preprocess data
2. Build and train the ANN model
3. Save model and preprocessing artifacts
4. Generate training visualizations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import yaml
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.sparse import save_npz, load_npz

from src.data_loader import DataLoader
from src.preprocessing import TextPreprocessor
from src.ann_model import DocumentClassifierANN


def setup_directories(config):
    """Create necessary directories."""
    paths = config['paths']
    for path in paths.values():
        Path(path).mkdir(parents=True, exist_ok=True)


def save_training_artifacts(ann, preprocessor, history, config, X_train, X_val, X_test, y_train, y_val, y_test):
    """Save all training artifacts."""
    paths = config['paths']

    # Save model
    ann.save_model()

    # Save preprocessing artifacts
    preprocessor.save_artifacts()

    # Save sparse matrices
    save_npz(Path(paths['data_processed']) / "X_train.npz", X_train)
    save_npz(Path(paths['data_processed']) / "X_val.npz", X_val)
    save_npz(Path(paths['data_processed']) / "X_test.npz", X_test)

    # Save labels
    np.save(Path(paths['data_processed']) / "y_train.npy", y_train)
    np.save(Path(paths['data_processed']) / "y_val.npy", y_val)
    np.save(Path(paths['data_processed']) / "y_test.npy", y_test)

    # Save training history
    history_path = Path(paths['results_metrics']) / "training_history.json"
    with open(history_path, 'w') as f:
        # Convert numpy types to native Python types
        history_serializable = {}
        for key, value in history.items():
            history_serializable[key] = [float(v) for v in value]
        json.dump(history_serializable, f, indent=2)
    print(f"Training history saved to {history_path}")

    # Save model configuration
    model_config_path = Path(paths['models']) / "model_config.yaml"
    with open(model_config_path, 'w') as f:
        yaml.dump(config['model'], f, default_flow_style=False)
    print(f"Model config saved to {model_config_path}")


def plot_training_curves(history, config):
    """Plot and save training curves."""
    paths = config['paths']
    results_figures = Path(paths['results_figures'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy curve
    axes[0].plot(history['accuracy'], label='Training Accuracy', linewidth=2, color='#2E86C1')
    axes[0].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='#E74C3C')
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])

    # Loss curve
    axes[1].plot(history['loss'], label='Training Loss', linewidth=2, color='#2E86C1')
    axes[1].plot(history['val_loss'], label='Validation Loss', linewidth=2, color='#E74C3C')
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = results_figures / "training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Training curves saved to {save_path}")
    plt.close()


def plot_category_distribution(df, config):
    """Plot category distribution."""
    paths = config['paths']
    results_figures = Path(paths['results_figures'])

    plt.figure(figsize=(12, 6))
    category_counts = df['category'].value_counts().sort_values(ascending=True)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(category_counts)))
    bars = plt.barh(range(len(category_counts)), category_counts.values, color=colors)
    plt.yticks(range(len(category_counts)), category_counts.index, fontsize=10)
    plt.xlabel('Number of Documents', fontsize=12)
    plt.title('Document Distribution Across Categories', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, category_counts.values)):
        plt.text(val + 50, bar.get_y() + bar.get_height()/2, str(val),
                va='center', fontsize=9)

    plt.tight_layout()
    save_path = results_figures / "category_distribution.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Category distribution saved to {save_path}")
    plt.close()


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("DOCUMENT CLASSIFICATION ANN - TRAINING PIPELINE")
    print("=" * 60)

    # Load configuration
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Setup directories
    setup_directories(config)

    # 1. Load dataset
    print("\n[1/6] Loading dataset...")
    loader = DataLoader()
    dataset = loader.load_dataset()
    df = loader.create_dataframe(dataset)

    # Plot category distribution
    plot_category_distribution(df, config)

    # 2. Split data
    print("\n[2/6] Splitting data...")
    train_df, val_df, test_df = loader.split_train_validation(df)

    # 3. Preprocess text
    print("\n[3/6] Preprocessing text...")
    preprocessor = TextPreprocessor()
    train_df = preprocessor.preprocess_dataframe(train_df)
    val_df = preprocessor.preprocess_dataframe(val_df)
    test_df = preprocessor.preprocess_dataframe(test_df)

    # 4. Vectorize text
    print("\n[4/6] Vectorizing text (TF-IDF)...")
    X_train = preprocessor.fit_transform(train_df['cleaned_text'].tolist())
    X_val = preprocessor.transform_texts(val_df['cleaned_text'].tolist())
    X_test = preprocessor.transform_texts(test_df['cleaned_text'].tolist())

    # 5. Encode labels
    print("\n[5/6] Encoding labels...")
    preprocessor.fit_label_encoder(train_df['category'].tolist())
    y_train = preprocessor.encode_labels(train_df['category'].tolist())
    y_val = preprocessor.encode_labels(val_df['category'].tolist())
    y_test = preprocessor.encode_labels(test_df['category'].tolist())

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Vocabulary size: {preprocessor.get_vocabulary_size()}")
    print(f"Number of classes: {len(preprocessor.label_encoder.classes_)}")

    # 6. Build and train model
    print("\n[6/6] Building and training ANN model...")
    ann = DocumentClassifierANN()
    actual_input_dim = X_train.shape[1]
    ann.build_model(input_dim=actual_input_dim)

    # Train
    history = ann.train(X_train, y_train, X_val, y_val)

    # Save all artifacts
    print("\nSaving artifacts...")
    save_training_artifacts(ann, preprocessor, history, config,
                           X_train, X_val, X_test, y_train, y_val, y_test)

    # Plot training curves
    plot_training_curves(history, config)

    # Print final results
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Best validation accuracy: {max(history['val_accuracy']):.4f}")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")
    print(f"Total epochs: {len(history['accuracy'])}")
    print(f"Model saved to: {config['paths']['models']}/document_classifier.keras")
    print(f"Vectorizer saved to: {config['paths']['models']}/tfidf_vectorizer.joblib")
    print(f"Label encoder saved to: {config['paths']['models']}/label_encoder.joblib")


if __name__ == "__main__":
    main()