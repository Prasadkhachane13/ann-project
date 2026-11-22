"""
Tests for Model Loading
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from pathlib import Path

from ann_model import DocumentClassifierANN
from preprocessing import TextPreprocessor


class TestModelLoading:
    """Test cases for model loading and artifacts."""

    def test_model_file_exists(self):
        """Test if model file exists."""
        model_path = Path("models/document_classifier.keras")
        # This test will pass if model exists, skip if not
        if not model_path.exists():
            pytest.skip("Model file not found. Run training first.")

    def test_vectorizer_file_exists(self):
        """Test if vectorizer file exists."""
        vectorizer_path = Path("models/tfidf_vectorizer.joblib")
        if not vectorizer_path.exists():
            pytest.skip("Vectorizer file not found. Run training first.")

    def test_label_encoder_file_exists(self):
        """Test if label encoder file exists."""
        encoder_path = Path("models/label_encoder.joblib")
        if not encoder_path.exists():
            pytest.skip("Label encoder file not found. Run training first.")

    def test_load_vectorizer(self):
        """Test loading vectorizer."""
        preprocessor = TextPreprocessor()
        if not preprocessor.load_artifacts():
            pytest.skip("Artifacts not found. Run training first.")

        assert preprocessor.vectorizer is not None
        assert preprocessor.label_encoder is not None
        assert preprocessor.get_vocabulary_size() > 0
        assert len(preprocessor.label_encoder.classes_) == 20

    def test_load_model(self):
        """Test loading ANN model."""
        ann = DocumentClassifierANN()
        if not ann.load_model():
            pytest.skip("Model not found. Run training first.")

        assert ann.model is not None
        assert ann.model.input_shape[1] > 0
        assert ann.model.output_shape[1] == 20

    def test_model_prediction_shape(self):
        """Test model prediction output shape."""
        preprocessor = TextPreprocessor()
        ann = DocumentClassifierANN()

        if not preprocessor.load_artifacts() or not ann.load_model():
            pytest.skip("Artifacts not found. Run training first.")

        # Create dummy input
        dummy_text = ["this is a test document for classification"]
        X = preprocessor.transform_texts(dummy_text)

        # Predict
        probs = ann.predict(X)
        assert probs.shape == (1, 20)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

        # Predict classes
        classes = ann.predict_classes(X)
        assert classes.shape == (1,)
        assert 0 <= classes[0] < 20

        # Predict top-k
        top_k_indices, top_k_probs = ann.predict_top_k(X, k=5)
        assert top_k_indices.shape == (1, 5)
        assert top_k_probs.shape == (1, 5)
        assert np.all(top_k_probs[0] >= 0)
        assert np.all(top_k_probs[0] <= 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])