"""
ANN Model Module for Document Classification

This module defines the Multilayer Perceptron (MLP) architecture for document classification.
"""

import yaml
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from tensorflow.keras.models import Model
from typing import Tuple, List, Optional, Dict
from pathlib import Path
import numpy as np


class DocumentClassifierANN:
    """Multilayer Perceptron for document classification."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize ANN model with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model_config = self.config['model']
        self.training_config = self.config['training']
        self.paths = self.config['paths']

        self.model: Optional[Model] = None
        self.history: Optional[Dict] = None
        self.model_path = Path(self.paths['models']) / "document_classifier.keras"

        # Set random seeds for reproducibility
        tf.random.set_seed(self.model_config['random_seed'])
        np.random.seed(self.model_config['random_seed'])

    def build_model(self, input_dim: int) -> Model:
        """
        Build the MLP model architecture.

        Args:
            input_dim: Number of input features (TF-IDF vocabulary size).

        Returns:
            Compiled Keras model.
        """
        print(f"Building model with input_dim={input_dim}")

        # Input layer
        inputs = layers.Input(shape=(input_dim,), name='input_layer')

        # Hidden layer 1
        x = layers.Dense(
            self.model_config['hidden_layers'][0]['units'],
            activation=self.model_config['hidden_layers'][0]['activation'],
            name='hidden_1'
        )(inputs)
        x = layers.Dropout(self.model_config['hidden_layers'][0]['dropout'], name='dropout_1')(x)

        # Hidden layer 2
        x = layers.Dense(
            self.model_config['hidden_layers'][1]['units'],
            activation=self.model_config['hidden_layers'][1]['activation'],
            name='hidden_2'
        )(x)
        x = layers.Dropout(self.model_config['hidden_layers'][1]['dropout'], name='dropout_2')(x)

        # Output layer
        outputs = layers.Dense(
            self.model_config['output_units'],
            activation=self.model_config['output_activation'],
            name='output_layer'
        )(x)

        # Create model
        self.model = Model(inputs=inputs, outputs=outputs, name='document_classifier_mlp')

        # Compile model
        optimizer = optimizers.Adam(learning_rate=self.model_config['learning_rate'])
        self.model.compile(
            optimizer=optimizer,
            loss=self.model_config['loss'],
            metrics=self.model_config['metrics']
        )

        print("Model built successfully.")
        self.model.summary()
        return self.model

    def get_callbacks(self) -> List[callbacks.Callback]:
        """Get training callbacks."""
        callback_list = [
            callbacks.EarlyStopping(
                monitor=self.model_config['early_stopping_monitor'],
                patience=self.model_config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath=str(self.model_path),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]
        return callback_list

    def train(self, X_train, y_train, X_val, y_val) -> Dict:
        """
        Train the model.

        Args:
            X_train: Training features (sparse matrix or array).
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.

        Returns:
            Training history dictionary.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        print("Starting model training...")

        # Convert sparse matrices to dense if needed (in batches for memory efficiency)
        # For now, we'll handle sparse input directly using TensorFlow's sparse tensor support
        # But Keras doesn't natively support sparse input for Dense layers
        # So we need to densify in batches or use a different approach

        # Check if input is sparse
        from scipy.sparse import issparse
        if issparse(X_train):
            print("Converting sparse matrices to dense for training...")
            print("Note: This may use significant memory. Consider using a sparse-aware approach for large datasets.")
            X_train_dense = X_train.toarray()
            X_val_dense = X_val.toarray()
        else:
            X_train_dense = X_train
            X_val_dense = X_val

        # Train model
        self.history = self.model.fit(
            X_train_dense, y_train,
            validation_data=(X_val_dense, y_val),
            batch_size=self.model_config['batch_size'],
            epochs=self.model_config['max_epochs'],
            callbacks=self.get_callbacks(),
            verbose=self.training_config['verbose'],
            shuffle=self.training_config['shuffle']
        )

        print("Training completed.")
        return self.history.history

    def evaluate(self, X_test, y_test, batch_size: int = None) -> Dict:
        """
        Evaluate the model on test data.

        Args:
            X_test: Test features.
            y_test: Test labels.
            batch_size: Batch size for evaluation.

        Returns:
            Dictionary with evaluation metrics.
        """
        if self.model is None:
            raise ValueError("Model not built or loaded.")

        if batch_size is None:
            batch_size = self.config['evaluation']['test_batch_size']

        from scipy.sparse import issparse
        if issparse(X_test):
            X_test_dense = X_test.toarray()
        else:
            X_test_dense = X_test

        results = self.model.evaluate(X_test_dense, y_test, batch_size=batch_size, verbose=1)
        metric_names = self.model.metrics_names
        return dict(zip(metric_names, results))

    def predict(self, X, batch_size: int = None) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input features.
            batch_size: Batch size for prediction.

        Returns:
            Array of prediction probabilities (n_samples, n_classes).
        """
        if self.model is None:
            raise ValueError("Model not built or loaded.")

        if batch_size is None:
            batch_size = self.config['evaluation']['test_batch_size']

        from scipy.sparse import issparse
        if issparse(X):
            X_dense = X.toarray()
        else:
            X_dense = X

        return self.model.predict(X_dense, batch_size=batch_size, verbose=0)

    def predict_classes(self, X, batch_size: int = None) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Input features.
            batch_size: Batch size for prediction.

        Returns:
            Array of predicted class indices.
        """
        probs = self.predict(X, batch_size)
        return np.argmax(probs, axis=1)

    def predict_top_k(self, X, k: int = 5, batch_size: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict top-k classes with probabilities.

        Args:
            X: Input features.
            k: Number of top predictions to return.
            batch_size: Batch size for prediction.

        Returns:
            Tuple of (top_k_indices, top_k_probabilities).
        """
        probs = self.predict(X, batch_size)
        top_k_indices = np.argsort(probs, axis=1)[:, -k:][:, ::-1]
        top_k_probs = np.take_along_axis(probs, top_k_indices, axis=1)
        return top_k_indices, top_k_probs

    def save_model(self, filepath: str = None) -> None:
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save.")

        if filepath is None:
            filepath = self.model_path

        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str = None) -> bool:
        """
        Load a trained model.

        Args:
            filepath: Path to model file.

        Returns:
            True if loaded successfully, False otherwise.
        """
        if filepath is None:
            filepath = self.model_path

        try:
            self.model = tf.keras.models.load_model(filepath)
            print(f"Model loaded from {filepath}")
            self.model.summary()
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def get_model_summary(self) -> str:
        """Get model summary as string."""
        if self.model is None:
            return "Model not built."
        import io
        stream = io.StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        return stream.getvalue()

    def plot_training_history(self, save_path: str = None) -> None:
        """Plot training history."""
        if self.history is None:
            print("No training history available.")
            return

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Accuracy
        axes[0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Loss
        axes[1].plot(self.history.history['loss'], label='Training Loss')
        axes[1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")

        plt.show()


def create_model_from_config(config_path: str = "config/config.yaml", input_dim: int = None) -> DocumentClassifierANN:
    """Factory function to create and build model from config."""
    ann = DocumentClassifierANN(config_path)
    if input_dim is None:
        input_dim = ann.model_config['input_dim']
    ann.build_model(input_dim)
    return ann


def main():
    """Test model creation."""
    ann = DocumentClassifierANN()
    model = ann.build_model(input_dim=15000)
    print("\nModel created successfully!")


if __name__ == "__main__":
    main()