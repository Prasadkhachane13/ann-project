"""
Data Loader Module for 20 Newsgroups Dataset

This module handles downloading, loading, and initial processing of the 20 Newsgroups dataset.
"""

import os
import pickle
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split


class DataLoader:
    """Handles loading and preprocessing of the 20 Newsgroups dataset."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize DataLoader with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.dataset_config = self.config['dataset']
        self.paths = self.config['paths']

        # Create directories
        Path(self.paths['data_raw']).mkdir(parents=True, exist_ok=True)
        Path(self.paths['data_processed']).mkdir(parents=True, exist_ok=True)

        # Cache file paths
        self.cache_dir = Path(self.paths['data_raw'])
        self.raw_cache_file = self.cache_dir / "20newsgroups_raw.pkl"
        self.processed_cache_file = Path(self.paths['data_processed']) / "20newsgroups_processed.csv"

    def load_dataset(self, force_download: bool = False) -> Dict:
        """
        Load the 20 Newsgroups dataset.

        Args:
            force_download: If True, re-download even if cached.

        Returns:
            Dictionary containing train/test data and metadata.
        """
        if not force_download and self.raw_cache_file.exists():
            print("Loading dataset from cache...")
            with open(self.raw_cache_file, 'rb') as f:
                return pickle.load(f)

        print("Downloading 20 Newsgroups dataset...")

        # Load training data
        train_data = fetch_20newsgroups(
            subset='train',
            categories=self.dataset_config['categories'],
            shuffle=self.dataset_config['shuffle'],
            random_state=self.dataset_config['random_state'],
            remove=self.dataset_config['remove']
        )

        # Load test data
        test_data = fetch_20newsgroups(
            subset='test',
            categories=self.dataset_config['categories'],
            shuffle=self.dataset_config['shuffle'],
            random_state=self.dataset_config['random_state'],
            remove=self.dataset_config['remove']
        )

        dataset = {
            'train_data': train_data.data,
            'train_target': train_data.target,
            'train_target_names': train_data.target_names,
            'test_data': test_data.data,
            'test_target': test_data.target,
            'test_target_names': test_data.target_names,
            'target_names': train_data.target_names
        }

        # Cache the raw dataset
        with open(self.raw_cache_file, 'wb') as f:
            pickle.dump(dataset, f)

        print(f"Dataset downloaded and cached. Training: {len(train_data.data)}, Test: {len(test_data.data)}")
        return dataset

    def create_dataframe(self, dataset: Dict) -> pd.DataFrame:
        """
        Create a structured DataFrame from the dataset.

        Args:
            dataset: Dictionary containing train/test data.

        Returns:
            DataFrame with document_id, text, category, category_id, cleaned_text, dataset_split.
        """
        # Combine train and test data
        all_texts = dataset['train_data'] + dataset['test_data']
        all_targets = list(dataset['train_target']) + list(dataset['test_target'])
        target_names = dataset['target_names']

        # Create splits
        n_train = len(dataset['train_data'])
        splits = ['train'] * n_train + ['test'] * (len(all_texts) - n_train)

        # Create DataFrame
        df = pd.DataFrame({
            'document_id': range(len(all_texts)),
            'document_text': all_texts,
            'category_id': all_targets,
            'category': [target_names[t] for t in all_targets],
            'dataset_split': splits
        })

        # Add cleaned text column (will be filled by preprocessing)
        df['cleaned_text'] = df['document_text']

        return df

    def split_train_validation(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split training data into train and validation sets with stratification.

        Args:
            df: Full DataFrame with dataset_split column.

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        train_df = df[df['dataset_split'] == 'train'].copy()
        test_df = df[df['dataset_split'] == 'test'].copy()

        # Split train into train/val with stratification
        val_size = self.dataset_config['validation_size']
        train_df, val_df = train_test_split(
            train_df,
            test_size=val_size,
            random_state=self.dataset_config['random_state'],
            stratify=train_df['category_id'] if self.dataset_config['stratify'] else None
        )

        train_df = train_df.copy()
        val_df = val_df.copy()
        train_df['dataset_split'] = 'train'
        val_df['dataset_split'] = 'validation'

        print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df

    def save_processed_data(self, df: pd.DataFrame) -> None:
        """Save processed DataFrame to CSV."""
        df.to_csv(self.processed_cache_file, index=False)
        print(f"Processed data saved to {self.processed_cache_file}")

    def load_processed_data(self) -> Optional[pd.DataFrame]:
        """Load processed DataFrame from CSV if exists."""
        if self.processed_cache_file.exists():
            return pd.read_csv(self.processed_cache_file)
        return None

    def get_category_info(self, dataset: Dict) -> pd.DataFrame:
        """Get category distribution information."""
        train_counts = pd.Series(dataset['train_target']).value_counts().sort_index()
        test_counts = pd.Series(dataset['test_target']).value_counts().sort_index()

        info = pd.DataFrame({
            'category_id': range(len(dataset['target_names'])),
            'category_name': dataset['target_names'],
            'train_count': train_counts.values,
            'test_count': test_counts.values,
            'total_count': train_counts.values + test_counts.values
        })
        return info

    def analyze_dataset(self, df: pd.DataFrame) -> Dict:
        """Perform basic dataset analysis."""
        analysis = {
            'total_documents': len(df),
            'num_categories': df['category_id'].nunique(),
            'categories': df['category'].unique().tolist(),
            'split_distribution': df['dataset_split'].value_counts().to_dict(),
            'category_distribution': df['category'].value_counts().to_dict(),
            'empty_documents': int((df['document_text'].str.strip() == '').sum()),
            'duplicate_documents': int(df['document_text'].duplicated().sum()),
            'avg_doc_length': df['document_text'].str.len().mean(),
            'median_doc_length': df['document_text'].str.len().median(),
            'min_doc_length': df['document_text'].str.len().min(),
            'max_doc_length': df['document_text'].str.len().max(),
        }
        return analysis


def main():
    """Main function to download and process dataset."""
    loader = DataLoader()

    # Load dataset
    dataset = loader.load_dataset()

    # Create DataFrame
    df = loader.create_dataframe(dataset)

    # Split train/validation
    train_df, val_df, test_df = loader.split_train_validation(df)

    # Combine back
    processed_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Save processed data
    loader.save_processed_data(processed_df)

    # Print analysis
    analysis = loader.analyze_dataset(processed_df)
    print("\nDataset Analysis:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")

    # Category info
    cat_info = loader.get_category_info(dataset)
    print("\nCategory Distribution:")
    print(cat_info.to_string(index=False))


if __name__ == "__main__":
    main()