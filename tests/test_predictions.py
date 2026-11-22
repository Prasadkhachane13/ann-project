"""
Tests for Prediction Module
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np

from predict import DocumentClassifier
from document_extractor import DocumentExtractor


class TestDocumentClassifier:
    """Test cases for DocumentClassifier."""

    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = DocumentClassifier()

    def test_classifier_initialization(self):
        """Test classifier initialization."""
        assert self.classifier is not None
        assert self.classifier.top_k == 5

    def test_model_loaded(self):
        """Test if model is loaded."""
        if not self.classifier._loaded:
            pytest.skip("Model not loaded. Run training first.")

    def test_get_categories(self):
        """Test getting categories."""
        if not self.classifier._loaded:
            pytest.skip("Model not loaded. Run training first.")

        categories = self.classifier.get_categories()
        assert isinstance(categories, list)
        assert len(categories) == 20
        assert all(isinstance(c, str) for c in categories)

        # Check some expected categories
        expected = ['comp.graphics', 'rec.sport.baseball', 'sci.space', 'talk.politics.guns']
        for cat in expected:
            assert cat in categories

    def test_predict_single(self):
        """Test single document prediction."""
        if not self.classifier._loaded:
            pytest.skip("Model not loaded. Run training first.")

        text = "The space shuttle launch was successful and the astronauts are conducting experiments."
        result = self.classifier.predict_single(text, "test_001")

        assert 'document_id' in result
        assert 'predicted_category' in result
        assert 'confidence' in result
        assert 'top_predictions' in result
        assert result['document_id'] == "test_001"
        assert isinstance(result['predicted_category'], str)
        assert 0 <= result['confidence'] <= 1
        assert len(result['top_predictions']) == 5

        # Check top predictions structure
        for i, pred in enumerate(result['top_predictions']):
            assert pred['rank'] == i + 1
            assert 'category' in pred
            assert 'probability' in pred
            assert 0 <= pred['probability'] <= 1

        # Check probabilities sum to ~1
        total_prob = sum(p['probability'] for p in result['top_predictions'])
        assert abs(total_prob - 1.0) < 0.01  # Top 5 should cover most probability mass

    def test_predict_batch(self):
        """Test batch document prediction."""
        if not self.classifier._loaded:
            pytest.skip("Model not loaded. Run training first.")

        texts = [
            "Baseball game ended with a home run",
            "Computer graphics rendering improved",
            "Medical research shows new treatment"
        ]
        doc_ids = ["batch_1", "batch_2", "batch_3"]

        results = self.classifier.predict_batch(texts, doc_ids)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result['document_id'] == doc_ids[i]
            assert 'predicted_category' in result
            assert 'confidence' in result
            assert 0 <= result['confidence'] <= 1

    def test_predict_batch_mismatched_ids(self):
        """Test batch prediction with mismatched IDs."""
        if not self.classifier._loaded:
            pytest.skip("Model not loaded. Run training first.")

        texts = ["text1", "text2"]
        doc_ids = ["id1"]  # Mismatched length

        with pytest.raises(ValueError, match="same length"):
            self.classifier.predict_batch(texts, doc_ids)

    def test_history_management(self):
        """Test prediction history."""
        if not self.classifier._loaded:
            pytest.skip("Model not loaded. Run training first.")

        # Clear history first
        self.classifier.clear_history()
        assert len(self.classifier.get_history()) == 0

        # Make a prediction
        self.classifier.predict_single("Test document for history", "history_test")

        # Check history
        history = self.classifier.get_history()
        assert len(history) >= 1
        last_entry = history[-1]
        assert last_entry['document_id'] == "history_test"
        assert 'predicted_category' in last_entry
        assert 'confidence' in last_entry
        assert 'timestamp' in last_entry

        # Test limit
        limited_history = self.classifier.get_history(limit=1)
        assert len(limited_history) == 1

        # Clear again
        self.classifier.clear_history()
        assert len(self.classifier.get_history()) == 0

    def test_get_model_info(self):
        """Test getting model info."""
        info = self.classifier.get_model_info()

        assert 'model_loaded' in info
        assert 'vocabulary_size' in info
        assert 'num_categories' in info
        assert 'categories' in info
        assert 'top_k' in info
        assert info['top_k'] == 5


class TestDocumentExtractor:
    """Test cases for DocumentExtractor."""

    def test_is_supported(self):
        """Test file extension support check."""
        assert DocumentExtractor.is_supported("test.txt") is True
        assert DocumentExtractor.is_supported("test.pdf") is True
        assert DocumentExtractor.is_supported("test.docx") is True
        assert DocumentExtractor.is_supported("test.doc") is False
        assert DocumentExtractor.is_supported("test.xyz") is False

    def test_get_extension(self):
        """Test getting file extension."""
        assert DocumentExtractor.get_extension("test.txt") == ".txt"
        assert DocumentExtractor.get_extension("TEST.PDF") == ".pdf"
        assert DocumentExtractor.get_extension("path/to/file.docx") == ".docx"

    def test_validate_file(self):
        """Test file validation."""
        # Valid file
        valid, error = DocumentExtractor.validate_file(b"content", "test.txt")
        assert valid is True
        assert error == ""

        # Empty file
        valid, error = DocumentExtractor.validate_file(b"", "test.txt")
        assert valid is False
        assert "empty" in error.lower()

        # Too large file
        large_content = b"x" * (11 * 1024 * 1024)  # 11 MB
        valid, error = DocumentExtractor.validate_file(large_content, "test.txt")
        assert valid is False
        assert "size" in error.lower()

        # Unsupported format
        valid, error = DocumentExtractor.validate_file(b"content", "test.xyz")
        assert valid is False
        assert "unsupported" in error.lower()

    def test_extract_txt(self):
        """Test TXT extraction."""
        content = b"Hello world\nThis is a test file."
        text, error = DocumentExtractor.extract_text(content, "test.txt")
        assert error is None
        assert "Hello world" in text
        assert "This is a test file" in text

    def test_extract_txt_encoding(self):
        """Test TXT extraction with different encodings."""
        # UTF-8
        content = "Hello café".encode('utf-8')
        text, error = DocumentExtractor.extract_text(content, "test.txt")
        assert error is None
        assert "café" in text

        # Latin-1
        content = "Hello café".encode('latin-1')
        text, error = DocumentExtractor.extract_text(content, "test.txt")
        assert error is None

    def test_extract_pdf_not_installed(self):
        """Test PDF extraction when PyMuPDF not installed."""
        # This test just checks the error handling
        content = b"%PDF-1.4 fake pdf content"
        text, error = DocumentExtractor.extract_text(content, "test.pdf")
        # Will either succeed or give a proper error message
        assert isinstance(text, str)
        assert isinstance(error, (str, type(None)))

    def test_extract_docx_not_installed(self):
        """Test DOCX extraction when python-docx not installed."""
        content = b"fake docx content"
        text, error = DocumentExtractor.extract_text(content, "test.docx")
        assert isinstance(text, str)
        assert isinstance(error, (str, type(None)))

    def test_extract_multiple(self):
        """Test multiple file extraction."""
        files = [
            (b"Text file 1", "file1.txt"),
            (b"Text file 2", "file2.txt"),
        ]
        results = DocumentExtractor.extract_from_multiple(files)

        assert len(results) == 2
        for result in results:
            assert 'filename' in result
            assert 'text' in result
            assert 'error' in result
            assert 'success' in result
            assert result['success'] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])