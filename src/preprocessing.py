"""
Text Preprocessing Module for Document Classification

This module handles text cleaning, normalization, and TF-IDF vectorization.
"""

import re
import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, save_npz, load_npz


class TextPreprocessor:
    """Handles text preprocessing and TF-IDF vectorization."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize TextPreprocessor with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.tfidf_config = self.config['tfidf']
        self.paths = self.config['paths']

        # Initialize components
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.label_encoder: Optional[LabelEncoder] = None

        # Create models directory
        Path(self.paths['models']).mkdir(parents=True, exist_ok=True)

        self.vectorizer_path = Path(self.paths['models']) / "tfidf_vectorizer.joblib"
        self.label_encoder_path = Path(self.paths['models']) / "label_encoder.joblib"

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Raw text string.

        Returns:
            Cleaned text string.
        """
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove email addresses
        text = re.sub(r'\S+@\S+\.\S+', '', text)

        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)

        # Remove newsgroup headers (lines starting with known headers)
        header_patterns = [
            r'^from:.*$',
            r'^subject:.*$',
            r'^organization:.*$',
            r'^lines:.*$',
            r'^nntp-posting-host:.*$',
            r'^message-id:.*$',
            r'^references:.*$',
            r'^date:.*$',
            r'^xref:.*$',
            r'^path:.*$',
            r'^newsgroups:.*$',
            r'^followup-to:.*$',
            r'^reply-to:.*$',
            r'^summary:.*$',
            r'^keywords:.*$',
            r'^distribution:.*$',
        ]
        for pattern in header_patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)

        # Remove quoted text (lines starting with > or |)
        text = re.sub(r'^[>|].*$', '', text, flags=re.MULTILINE)

        # Remove multiple spaces and normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'document_text',
                             cleaned_column: str = 'cleaned_text') -> pd.DataFrame:
        """
        Apply text cleaning to a DataFrame column.

        Args:
            df: Input DataFrame.
            text_column: Name of column containing raw text.
            cleaned_column: Name of column to store cleaned text.

        Returns:
            DataFrame with cleaned text column added/updated.
        """
        df = df.copy()
        print(f"Cleaning {len(df)} documents...")
        df[cleaned_column] = df[text_column].apply(self.clean_text)

        # Handle empty documents after cleaning
        empty_mask = df[cleaned_column].str.strip() == ''
        if empty_mask.any():
            print(f"Warning: {empty_mask.sum()} documents became empty after cleaning")
            # Replace empty with a placeholder
            df.loc[empty_mask, cleaned_column] = "empty document"

        return df

    def create_vectorizer(self) -> TfidfVectorizer:
        """Create and configure TF-IDF vectorizer."""
        return TfidfVectorizer(
            max_features=self.tfidf_config['max_features'],
            ngram_range=tuple(self.tfidf_config['ngram_range']),
            min_df=self.tfidf_config['min_df'],
            max_df=self.tfidf_config['max_df'],
            sublinear_tf=self.tfidf_config['sublinear_tf'],
            stop_words=self.tfidf_config['stop_words'],
            dtype=np.dtype(self.tfidf_config['dtype']),
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\b[a-zA-Z]{2,}\b'  # Only words with 2+ letters
        )

    def fit_vectorizer(self, texts: List[str]) -> TfidfVectorizer:
        """
        Fit TF-IDF vectorizer on training texts.

        Args:
            texts: List of cleaned text strings.

        Returns:
            Fitted TfidfVectorizer.
        """
        print(f"Fitting TF-IDF vectorizer on {len(texts)} documents...")
        self.vectorizer = self.create_vectorizer()
        self.vectorizer.fit(texts)
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        return self.vectorizer

    def transform_texts(self, texts: List[str]) -> csr_matrix:
        """
        Transform texts to TF-IDF vectors.

        Args:
            texts: List of cleaned text strings.

        Returns:
            Sparse TF-IDF matrix.
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit_vectorizer() first.")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: List[str]) -> csr_matrix:
        """
        Fit vectorizer and transform texts in one step.

        Args:
            texts: List of cleaned text strings.

        Returns:
            Sparse TF-IDF matrix.
        """
        self.fit_vectorizer(texts)
        return self.transform_texts(texts)

    def fit_label_encoder(self, labels: List[str]) -> LabelEncoder:
        """
        Fit label encoder on category labels.

        Args:
            labels: List of category names.

        Returns:
            Fitted LabelEncoder.
        """
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(labels)
        print(f"Label encoder fitted with {len(self.label_encoder.classes_)} classes")
        return self.label_encoder

    def encode_labels(self, labels: List[str]) -> np.ndarray:
        """
        Encode category labels to integers.

        Args:
            labels: List of category names.

        Returns:
            Array of encoded labels.
        """
        if self.label_encoder is None:
            raise ValueError("Label encoder not fitted. Call fit_label_encoder() first.")
        return self.label_encoder.transform(labels)

    def decode_labels(self, encoded_labels: np.ndarray) -> np.ndarray:
        """
        Decode integer labels to category names.

        Args:
            encoded_labels: Array of encoded labels.

        Returns:
            Array of category names.
        """
        if self.label_encoder is None:
            raise ValueError("Label encoder not fitted.")
        return self.label_encoder.inverse_transform(encoded_labels)

    def save_artifacts(self) -> None:
        """Save vectorizer and label encoder to disk."""
        if self.vectorizer is not None:
            joblib.dump(self.vectorizer, self.vectorizer_path)
            print(f"Vectorizer saved to {self.vectorizer_path}")

        if self.label_encoder is not None:
            joblib.dump(self.label_encoder, self.label_encoder_path)
            print(f"Label encoder saved to {self.label_encoder_path}")

    def load_artifacts(self) -> bool:
        """
        Load vectorizer and label encoder from disk.

        Returns:
            True if both loaded successfully, False otherwise.
        """
        try:
            if self.vectorizer_path.exists():
                self.vectorizer = joblib.load(self.vectorizer_path)
                print(f"Vectorizer loaded from {self.vectorizer_path}")
            else:
                print(f"Vectorizer not found at {self.vectorizer_path}")
                return False

            if self.label_encoder_path.exists():
                self.label_encoder = joblib.load(self.label_encoder_path)
                print(f"Label encoder loaded from {self.label_encoder_path}")
            else:
                print(f"Label encoder not found at {self.label_encoder_path}")
                return False

            return True
        except Exception as e:
            print(f"Error loading artifacts: {e}")
            return False

    def save_sparse_matrix(self, matrix: csr_matrix, filepath: Union[str, Path]) -> None:
        """Save sparse matrix to disk."""
        save_npz(filepath, matrix)
        print(f"Sparse matrix saved to {filepath}")

    def load_sparse_matrix(self, filepath: Union[str, Path]) -> csr_matrix:
        """Load sparse matrix from disk."""
        return load_npz(filepath)

    def get_feature_names(self) -> np.ndarray:
        """Get feature names from vectorizer."""
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted.")
        return self.vectorizer.get_feature_names_out()

    def get_vocabulary_size(self) -> int:
        """Get vocabulary size."""
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted.")
        return len(self.vectorizer.vocabulary_)


def main():
    """Test preprocessing module."""
    from src.data_loader import DataLoader

    # Load data
    loader = DataLoader()
    dataset = loader.load_dataset()
    df = loader.create_dataframe(dataset)
    train_df, val_df, test_df = loader.split_train_validation(df)

    # Initialize preprocessor
    preprocessor = TextPreprocessor()

    # Clean texts
    train_df = preprocessor.preprocess_dataframe(train_df)
    val_df = preprocessor.preprocess_dataframe(val_df)
    test_df = preprocessor.preprocess_dataframe(test_df)

    # Fit vectorizer on training data only
    X_train = preprocessor.fit_transform(train_df['cleaned_text'].tolist())
    X_val = preprocessor.transform_texts(val_df['cleaned_text'].tolist())
    X_test = preprocessor.transform_texts(test_df['cleaned_text'].tolist())

    # Fit label encoder
    preprocessor.fit_label_encoder(train_df['category'].tolist())
    y_train = preprocessor.encode_labels(train_df['category'].tolist())
    y_val = preprocessor.encode_labels(val_df['category'].tolist())
    y_test = preprocessor.encode_labels(test_df['category'].tolist())

    # Save artifacts
    preprocessor.save_artifacts()

    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Number of classes: {len(preprocessor.label_encoder.classes_)}")


if __name__ == "__main__":
    main()