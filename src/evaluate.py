"""
Evaluation Script for Document Classification ANN

This script evaluates the trained model on the test dataset and generates
comprehensive evaluation metrics and visualizations.
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
from scipy.sparse import load_npz
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

from src.data_loader import DataLoader
from src.preprocessing import TextPreprocessor
from src.ann_model import DocumentClassifierANN


def load_test_data(config):
    """Load test data and model artifacts."""
    paths = config['paths']

    # Load processed data
    X_test = load_npz(Path(paths['data_processed']) / "X_test.npz")
    y_test = np.load(Path(paths['data_processed']) / "y_test.npy")

    # Load preprocessing artifacts
    preprocessor = TextPreprocessor()
    if not preprocessor.load_artifacts():
        raise FileNotFoundError("Preprocessing artifacts not found. Run training first.")

    # Load model
    ann = DocumentClassifierANN()
    if not ann.load_model():
        raise FileNotFoundError("Model not found. Run training first.")

    # Load category names
    loader = DataLoader()
    dataset = loader.load_dataset()
    target_names = dataset['target_names']

    return X_test, y_test, preprocessor, ann, target_names


def compute_metrics(y_true, y_pred, y_probs, target_names):
    """Compute comprehensive evaluation metrics."""
    metrics = {}

    # Basic metrics
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['precision_weighted'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
    metrics['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['recall_weighted'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
    metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))

    # Per-class metrics
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    metrics['per_class'] = report

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()

    # Top-k accuracy (k=5)
    top5_preds = np.argsort(y_probs, axis=1)[:, -5:][:, ::-1]
    top5_correct = np.any(top5_preds == y_true.reshape(-1, 1), axis=1)
    metrics['top5_accuracy'] = float(np.mean(top5_correct))

    return metrics


def save_evaluation_results(metrics, config, target_names):
    """Save evaluation metrics and reports."""
    paths = config['paths']
    results_metrics = Path(paths['results_metrics'])
    results_reports = Path(paths['results_reports'])
    results_figures = Path(paths['results_figures'])

    # Save metrics as JSON
    metrics_path = results_metrics / "evaluation_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Save classification report as CSV
    report_df = pd.DataFrame(metrics['per_class']).T
    report_path = results_reports / "classification_report.csv"
    report_df.to_csv(report_path, float_format='%.4f')
    print(f"Classification report saved to {report_path}")

    # Save classification report as text
    report_text_path = results_reports / "classification_report.txt"
    with open(report_text_path, 'w') as f:
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Overall Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Macro Precision: {metrics['precision_macro']:.4f}\n")
        f.write(f"Weighted Precision: {metrics['precision_weighted']:.4f}\n")
        f.write(f"Macro Recall: {metrics['recall_macro']:.4f}\n")
        f.write(f"Weighted Recall: {metrics['recall_weighted']:.4f}\n")
        f.write(f"Macro F1-Score: {metrics['f1_macro']:.4f}\n")
        f.write(f"Weighted F1-Score: {metrics['f1_weighted']:.4f}\n")
        f.write(f"Top-5 Accuracy: {metrics['top5_accuracy']:.4f}\n\n")
        f.write("PER-CLASS METRICS:\n")
        f.write("-" * 60 + "\n")
        report_df.to_string(f, float_format='%.4f')

    print(f"Text report saved to {report_text_path}")

    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(metrics['confusion_matrix'],
                         index=target_names, columns=target_names)
    cm_path = results_reports / "confusion_matrix.csv"
    cm_df.to_csv(cm_path)
    print(f"Confusion matrix saved to {cm_path}")


def plot_confusion_matrix(metrics, target_names, config):
    """Plot and save confusion matrix."""
    paths = config['paths']
    results_figures = Path(paths['results_figures'])

    cm = np.array(metrics['confusion_matrix'])

    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Raw confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45, labelsize=9)
    axes[0].tick_params(axis='y', rotation=0, labelsize=9)

    # Normalized confusion matrix
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names,
                ax=axes[1], cbar_kws={'label': 'Proportion'})
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45, labelsize=9)
    axes[1].tick_params(axis='y', rotation=0, labelsize=9)

    plt.tight_layout()
    save_path = results_figures / "confusion_matrix.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Confusion matrix plot saved to {save_path}")
    plt.close()


def plot_per_class_metrics(metrics, target_names, config):
    """Plot per-class precision, recall, F1-score."""
    paths = config['paths']
    results_figures = Path(paths['results_figures'])

    report = metrics['per_class']

    # Extract per-class metrics
    classes = target_names
    precision = [report[c]['precision'] for c in classes]
    recall = [report[c]['recall'] for c in classes]
    f1 = [report[c]['f1-score'] for c in classes]
    support = [report[c]['support'] for c in classes]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Precision
    bars1 = axes[0, 0].barh(range(len(classes)), precision, color='#2E86C1', alpha=0.8)
    axes[0, 0].set_yticks(range(len(classes)))
    axes[0, 0].set_yticklabels(classes, fontsize=9)
    axes[0, 0].set_xlabel('Precision', fontsize=11)
    axes[0, 0].set_title('Per-Class Precision', fontsize=13, fontweight='bold')
    axes[0, 0].set_xlim([0, 1.05])
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    for i, (bar, val) in enumerate(zip(bars1, precision)):
        axes[0, 0].text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                       va='center', fontsize=8)

    # Recall
    bars2 = axes[0, 1].barh(range(len(classes)), recall, color='#27AE60', alpha=0.8)
    axes[0, 1].set_yticks(range(len(classes)))
    axes[0, 1].set_yticklabels(classes, fontsize=9)
    axes[0, 1].set_xlabel('Recall', fontsize=11)
    axes[0, 1].set_title('Per-Class Recall', fontsize=13, fontweight='bold')
    axes[0, 1].set_xlim([0, 1.05])
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    for i, (bar, val) in enumerate(zip(bars2, recall)):
        axes[0, 1].text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                       va='center', fontsize=8)

    # F1-Score
    bars3 = axes[1, 0].barh(range(len(classes)), f1, color='#E74C3C', alpha=0.8)
    axes[1, 0].set_yticks(range(len(classes)))
    axes[1, 0].set_yticklabels(classes, fontsize=9)
    axes[1, 0].set_xlabel('F1-Score', fontsize=11)
    axes[1, 0].set_title('Per-Class F1-Score', fontsize=13, fontweight='bold')
    axes[1, 0].set_xlim([0, 1.05])
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    for i, (bar, val) in enumerate(zip(bars3, f1)):
        axes[1, 0].text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                       va='center', fontsize=8)

    # Support
    bars4 = axes[1, 1].barh(range(len(classes)), support, color='#8E44AD', alpha=0.8)
    axes[1, 1].set_yticks(range(len(classes)))
    axes[1, 1].set_yticklabels(classes, fontsize=9)
    axes[1, 1].set_xlabel('Support (Test Samples)', fontsize=11)
    axes[1, 1].set_title('Per-Class Support', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    for i, (bar, val) in enumerate(zip(bars4, support)):
        axes[1, 1].text(val + 1, bar.get_y() + bar.get_height()/2, str(val),
                       va='center', fontsize=8)

    plt.tight_layout()
    save_path = results_figures / "per_class_metrics.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Per-class metrics plot saved to {save_path}")
    plt.close()


def plot_training_history(config):
    """Plot training history from saved JSON."""
    paths = config['paths']
    results_metrics = Path(paths['results_metrics'])
    results_figures = Path(paths['results_figures'])

    history_path = results_metrics / "training_history.json"
    if not history_path.exists():
        print("Training history not found.")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['accuracy']) + 1)

    axes[0].plot(epochs, history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    axes[0].plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[0].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])

    axes[1].plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
    axes[1].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[1].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = results_figures / "training_history.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Training history plot saved to {save_path}")
    plt.close()


def main():
    """Main evaluation pipeline."""
    print("=" * 60)
    print("DOCUMENT CLASSIFICATION ANN - EVALUATION PIPELINE")
    print("=" * 60)

    # Load configuration
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Load test data and model
    print("\nLoading test data and model...")
    X_test, y_test, preprocessor, ann, target_names = load_test_data(config)

    # Predict
    print("\nRunning predictions on test set...")
    y_probs = ann.predict(X_test)
    y_pred = np.argmax(y_probs, axis=1)

    # Compute metrics
    print("\nComputing evaluation metrics...")
    metrics = compute_metrics(y_test, y_pred, y_probs, target_names)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Test Accuracy:     {metrics['accuracy']:.4f}")
    print(f"Top-5 Accuracy:    {metrics['top5_accuracy']:.4f}")
    print(f"Macro Precision:   {metrics['precision_macro']:.4f}")
    print(f"Weighted Precision: {metrics['precision_weighted']:.4f}")
    print(f"Macro Recall:      {metrics['recall_macro']:.4f}")
    print(f"Weighted Recall:   {metrics['recall_weighted']:.4f}")
    print(f"Macro F1-Score:    {metrics['f1_macro']:.4f}")
    print(f"Weighted F1-Score: {metrics['f1_weighted']:.4f}")

    # Save results
    print("\nSaving evaluation results...")
    save_evaluation_results(metrics, config, target_names)

    # Generate plots
    print("\nGenerating visualizations...")
    plot_confusion_matrix(metrics, target_names, config)
    plot_per_class_metrics(metrics, target_names, config)
    plot_training_history(config)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()