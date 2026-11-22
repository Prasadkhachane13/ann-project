"""
Prediction Module for Document Classification ANN

This module provides inference capabilities for the trained model,
including single and batch document classification.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union

from src.preprocessing import TextPreprocessor
from src.ann_model import DocumentClassifierANN


class DocumentClassifier:
    """High-level interface for document classification."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize classifier with saved model and preprocessing artifacts."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.paths = self.config['paths']
        self.top_k = self.config['evaluation']['top_k_predictions']

        # Load artifacts
        self.preprocessor = TextPreprocessor(config_path)
        self.ann = DocumentClassifierANN(config_path)

        self._loaded = False
        self._load_artifacts()

        # Prediction history
        self.history_file = Path(self.paths['data_processed']) / "prediction_history.json"
        self.history = self._load_history()

    def _load_artifacts(self) -> bool:
        """Load model and preprocessing artifacts."""
        preprocessor_loaded = self.preprocessor.load_artifacts()
        model_loaded = self.ann.load_model()

        self._loaded = preprocessor_loaded and model_loaded
        return self._loaded

    def _load_history(self) -> List[Dict]:
        """Load prediction history from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_history(self) -> None:
        """Save prediction history to file."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save prediction history: {e}")

    def _add_to_history(self, doc_id: str, predicted_category: str,
                        confidence: float, top_predictions: List[Dict]) -> None:
        """Add prediction to history."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'document_id': doc_id,
            'predicted_category': predicted_category,
            'confidence': float(confidence),
            'top_predictions': top_predictions
        }
        self.history.append(entry)

        # Limit history size
        max_history = self.config['app']['prediction_history_limit']
        if len(self.history) > max_history:
            self.history = self.history[-max_history:]

        self._save_history()

    def predict_single(self, text: str, doc_id: str = None) -> Dict:
        """
        Classify a single document.

        Args:
            text: Document text to classify.
            doc_id: Optional document identifier.

        Returns:
            Dictionary with prediction results.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Check if artifacts exist.")

        if doc_id is None:
            doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Preprocess text
        cleaned_text = self.preprocessor.clean_text(text)

        # Vectorize
        X = self.preprocessor.transform_texts([cleaned_text])

        # Predict
        top_k_indices, top_k_probs = self.ann.predict_top_k(X, k=self.top_k)

        # Decode predictions
        predicted_idx = top_k_indices[0, 0]
        predicted_category = self.preprocessor.decode_labels([predicted_idx])[0]
        confidence = float(top_k_probs[0, 0])

        # Top-k predictions
        top_predictions = []
        for i in range(self.top_k):
            idx = top_k_indices[0, i]
            prob = top_k_probs[0, i]
            category = self.preprocessor.decode_labels([idx])[0]
            top_predictions.append({
                'category': category,
                'probability': float(prob),
                'rank': i + 1
            })

        # Add to history
        self._add_to_history(doc_id, predicted_category, confidence, top_predictions)

        return {
            'document_id': doc_id,
            'predicted_category': predicted_category,
            'confidence': confidence,
            'top_predictions': top_predictions,
            'cleaned_text_preview': cleaned_text[:200] + "..." if len(cleaned_text) > 200 else cleaned_text
        }

    def predict_batch(self, texts: List[str], doc_ids: List[str] = None) -> List[Dict]:
        """
        Classify multiple documents.

        Args:
            texts: List of document texts.
            doc_ids: Optional list of document identifiers.

        Returns:
            List of prediction result dictionaries.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Check if artifacts exist.")

        if doc_ids is None:
            doc_ids = [f"doc_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}" for i in range(len(texts))]

        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must have the same length")

        # Preprocess all texts
        cleaned_texts = [self.preprocessor.clean_text(text) for text in texts]

        # Vectorize
        X = self.preprocessor.transform_texts(cleaned_texts)

        # Predict
        top_k_indices, top_k_probs = self.ann.predict_top_k(X, k=self.top_k)

        # Process results
        results = []
        for i in range(len(texts)):
            predicted_idx = top_k_indices[i, 0]
            predicted_category = self.preprocessor.decode_labels([predicted_idx])[0]
            confidence = float(top_k_probs[i, 0])

            top_predictions = []
            for k in range(self.top_k):
                idx = top_k_indices[i, k]
                prob = top_k_probs[i, k]
                category = self.preprocessor.decode_labels([idx])[0]
                top_predictions.append({
                    'category': category,
                    'probability': float(prob),
                    'rank': k + 1
                })

            result = {
                'document_id': doc_ids[i],
                'predicted_category': predicted_category,
                'confidence': confidence,
                'top_predictions': top_predictions,
                'cleaned_text_preview': cleaned_texts[i][:200] + "..." if len(cleaned_texts[i]) > 200 else cleaned_texts[i]
            }
            results.append(result)

            # Add to history
            self._add_to_history(doc_ids[i], predicted_category, confidence, top_predictions)

        return results

    def get_categories(self) -> List[str]:
        """Get list of all categories."""
        if self.preprocessor.label_encoder is None:
            raise RuntimeError("Label encoder not loaded.")
        return self.preprocessor.label_encoder.classes_.tolist()

    def get_history(self, limit: int = None) -> List[Dict]:
        """Get prediction history."""
        if limit is None:
            return self.history
        return self.history[-limit:]

    def clear_history(self) -> None:
        """Clear prediction history."""
        self.history = []
        self._save_history()

    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'model_loaded': self._loaded,
            'vocabulary_size': self.preprocessor.get_vocabulary_size() if self.preprocessor.vectorizer else 0,
            'num_categories': len(self.get_categories()) if self._loaded else 0,
            'categories': self.get_categories() if self._loaded else [],
            'top_k': self.top_k,
            'model_path': str(Path(self.paths['models']) / "document_classifier.keras"),
            'vectorizer_path': str(Path(self.paths['models']) / "tfidf_vectorizer.joblib"),
        }


def main():
    """Test prediction module."""
    print("Testing Document Classifier...")

    classifier = DocumentClassifier()

    if not classifier._loaded:
        print("Model not loaded. Please run training first.")
        return

    # Test single prediction
    test_text = """
    The space shuttle launch was delayed due to technical issues with the main engines.
    NASA engineers are working to resolve the problem before the next launch window.
    The mission to the international space station is critical for ongoing research.
    """

    print("\nSingle Document Classification Test:")
    result = classifier.predict_single(test_text, "test_001")
    print(f"Document ID: {result['document_id']}")
    print(f"Predicted Category: {result['predicted_category']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("Top 5 Predictions:")
    for pred in result['top_predictions']:
        print(f"  {pred['rank']}. {pred['category']}: {pred['probability']:.4f}")

    # Test batch prediction
    test_texts = [
        "The baseball game ended with a home run in the final inning.",
        "Computer graphics rendering techniques have improved significantly.",
        "Medical researchers discovered a new treatment for the disease."
    ]

    print("\nBatch Classification Test:")
    results = classifier.predict_batch(test_texts, ["test_002", "test_003", "test_004"])
    for result in results:
        print(f"  {result['document_id']}: {result['predicted_category']} ({result['confidence']:.4f})")

    # Print model info
    print("\nModel Info:")
    info = classifier.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()