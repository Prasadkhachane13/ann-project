"""
Tests for Preprocessing Module
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from preprocessing import TextPreprocessor


class TestTextPreprocessor:
    """Test cases for TextPreprocessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = TextPreprocessor("config/config.yaml")

    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        text = "  Hello WORLD!  This is a TEST.  "
        cleaned = self.preprocessor.clean_text(text)
        assert cleaned == "hello world! this is a test."

    def test_clean_text_email_removal(self):
        """Test email address removal."""
        text = "Contact me at john@example.com for more info."
        cleaned = self.preprocessor.clean_text(text)
        assert "john@example.com" not in cleaned
        assert "contact me at for more info" in cleaned

    def test_clean_text_url_removal(self):
        """Test URL removal."""
        text = "Visit http://example.com or https://test.org/page"
        cleaned = self.preprocessor.clean_text(text)
        assert "http://example.com" not in cleaned
        assert "https://test.org/page" not in cleaned

    def test_clean_text_headers(self):
        """Test newsgroup header removal."""
        text = """From: user@example.com
Subject: Test message
Organization: Test Org

This is the actual message body."""
        cleaned = self.preprocessor.clean_text(text)
        assert "from:" not in cleaned.lower()
        assert "subject:" not in cleaned.lower()
        assert "organization:" not in cleaned.lower()
        assert "this is the actual message body" in cleaned

    def test_clean_text_quoted(self):
        """Test quoted text removal."""
        text = """This is my response.
> This is quoted text
> Another quoted line
My response continues."""
        cleaned = self.preprocessor.clean_text(text)
        assert "> this is quoted text" not in cleaned
        assert "> another quoted line" not in cleaned
        assert "this is my response" in cleaned
        assert "my response continues" in cleaned

    def test_clean_text_empty(self):
        """Test handling of empty/whitespace text."""
        assert self.preprocessor.clean_text("") == ""
        assert self.preprocessor.clean_text("   ") == ""
        assert self.preprocessor.clean_text("\n\t\r") == ""

    def test_clean_text_non_string(self):
        """Test handling of non-string input."""
        assert self.preprocessor.clean_text(None) == ""
        assert self.preprocessor.clean_text(123) == ""
        assert self.preprocessor.clean_text([]) == ""

    def test_preprocess_dataframe(self):
        """Test DataFrame preprocessing."""
        df = pd.DataFrame({
            'document_text': [
                "Hello WORLD!",
                "  Test   message  ",
                "Email: test@example.com"
            ],
            'category': ['cat1', 'cat2', 'cat1']
        })

        result = self.preprocessor.preprocess_dataframe(df)

        assert 'cleaned_text' in result.columns
        assert len(result) == 3
        assert result['cleaned_text'].iloc[0] == "hello world!"
        assert result['cleaned_text'].iloc[1] == "test message"
        assert "test@example.com" not in result['cleaned_text'].iloc[2]

    def test_create_vectorizer(self):
        """Test vectorizer creation."""
        vectorizer = self.preprocessor.create_vectorizer()
        assert isinstance(vectorizer, type(self.preprocessor.vectorizer)) or vectorizer is not None
        assert vectorizer.max_features == 15000
        assert vectorizer.ngram_range == (1, 2)
        assert vectorizer.min_df == 2
        assert vectorizer.max_df == 0.95

    def test_fit_transform(self):
        """Test fit and transform."""
        texts = [
            "this is a test document",
            "another test document here",
            "completely different content"
        ]

        X = self.preprocessor.fit_transform(texts)

        assert isinstance(X, csr_matrix)
        assert X.shape[0] == 3
        assert X.shape[1] <= 15000
        assert self.preprocessor.vectorizer is not None
        assert len(self.preprocessor.vectorizer.vocabulary_) > 0

    def test_transform_without_fit(self):
        """Test transform without fitting raises error."""
        with pytest.raises(ValueError, match="Vectorizer not fitted"):
            self.preprocessor.transform_texts(["test"])

    def test_label_encoder(self):
        """Test label encoder."""
        labels = ['cat1', 'cat2', 'cat1', 'cat3']
        self.preprocessor.fit_label_encoder(labels)

        encoded = self.preprocessor.encode_labels(['cat1', 'cat2', 'cat3'])
        assert list(encoded) == [0, 1, 2]

        decoded = self.preprocessor.decode_labels(encoded)
        assert list(decoded) == ['cat1', 'cat2', 'cat3']

    def test_encode_without_fit(self):
        """Test encode without fitting raises error."""
        preprocessor = TextPreprocessor("config/config.yaml")
        with pytest.raises(ValueError, match="Label encoder not fitted"):
            preprocessor.encode_labels(['cat1'])

    def test_get_feature_names(self):
        """Test getting feature names."""
        texts = ["test document", "another test"]
        self.preprocessor.fit_transform(texts)
        features = self.preprocessor.get_feature_names()
        assert isinstance(features, np.ndarray)
        assert len(features) > 0

    def test_vocabulary_size(self):
        """Test vocabulary size."""
        texts = ["test document", "another test"]
        self.preprocessor.fit_transform(texts)
        size = self.preprocessor.get_vocabulary_size()
        assert isinstance(size, int)
        assert size > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])