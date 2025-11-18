"""
Meta-Learning for Constraint Weights

This module implements meta-learning to optimize constraint weights based on
validation set performance. Instead of using fixed empirical frequencies,
we learn which constraints are most predictive of correct solutions.

Key components:
1. Dataset Collection: Generate candidates and extract constraint features
2. Weight Learning: Optimize weights using validation data
3. Evaluation: Test learned weights on held-out tasks

Learning approaches:
- Logistic Regression: Fast, interpretable
- Grid Search: Exhaustive but slow
- Gradient Descent: Flexible, allows complex scoring functions
"""

import numpy as np
import json
import pickle
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import torch

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Only gradient descent method will work.")

import arc_constraints


class ConstraintFeatureExtractor:
    """
    Extracts constraint satisfaction features from candidate solutions.
    """

    def __init__(self, validator: arc_constraints.ARCConstraintValidator):
        """
        Initialize feature extractor with a constraint validator.

        Args:
            validator: ARCConstraintValidator instance for the task
        """
        self.validator = validator
        self.constraint_names = [
            'color_preservation',
            'non_trivial',
            'size_consistency',
            'background_consistency',
            'color_count',
            'symmetry_preservation'
        ]

    def extract_features(self, test_input_grid: np.ndarray,
                        candidate_output: List[List[int]]) -> Dict[str, float]:
        """
        Extract constraint features from a candidate solution.

        Args:
            test_input_grid: Input grid for test example
            candidate_output: Predicted output grid

        Returns:
            Dictionary mapping constraint names to binary features (0 or 1)
        """
        output_array = np.array(candidate_output, dtype=int)

        # Map colors back to original indices
        color_mapping = {self.validator.task.colors[i]: i
                        for i in range(len(self.validator.task.colors))}
        input_remapped = np.array([[color_mapping.get(int(val), int(val))
                                   for val in row] for row in test_input_grid], dtype=int)
        output_remapped = np.array([[color_mapping.get(int(val), int(val))
                                    for val in row] for row in output_array], dtype=int)

        # Extract all constraint features
        features = {}

        is_valid, _ = self.validator.validate_color_preservation(input_remapped, output_remapped)
        features['color_preservation'] = float(is_valid)

        is_valid, _ = self.validator.validate_non_trivial(output_remapped)
        features['non_trivial'] = float(is_valid)

        is_valid, _ = self.validator.validate_size_consistency(input_remapped, output_remapped)
        features['size_consistency'] = float(is_valid)

        is_valid, _ = self.validator.validate_background_consistency(input_remapped, output_remapped)
        features['background_consistency'] = float(is_valid)

        is_valid, _ = self.validator.validate_color_count(input_remapped, output_remapped)
        features['color_count'] = float(is_valid)

        is_valid, _ = self.validator.validate_symmetry_preservation(input_remapped, output_remapped)
        features['symmetry_preservation'] = float(is_valid)

        return features


class MetaLearningDataset:
    """
    Dataset for meta-learning constraint weights.

    Stores (constraint_features, is_correct_label) pairs from validation tasks.
    """

    def __init__(self):
        self.features = []  # List of feature dicts
        self.labels = []    # List of binary labels (correct/incorrect)
        self.task_ids = []  # Track which task each sample came from
        self.constraint_names = [
            'color_preservation',
            'non_trivial',
            'size_consistency',
            'background_consistency',
            'color_count',
            'symmetry_preservation'
        ]

    def add_sample(self, features: Dict[str, float], is_correct: bool, task_id: str):
        """Add a training sample to the dataset."""
        self.features.append(features)
        self.labels.append(1.0 if is_correct else 0.0)
        self.task_ids.append(task_id)

    def get_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get dataset as numpy arrays.

        Returns:
            X: (n_samples, n_constraints) feature matrix
            y: (n_samples,) label vector
        """
        n_samples = len(self.features)
        n_constraints = len(self.constraint_names)

        X = np.zeros((n_samples, n_constraints))
        y = np.array(self.labels)

        for i, feature_dict in enumerate(self.features):
            for j, name in enumerate(self.constraint_names):
                X[i, j] = feature_dict.get(name, 0.0)

        return X, y

    def save(self, filepath: str):
        """Save dataset to file."""
        data = {
            'features': self.features,
            'labels': self.labels,
            'task_ids': self.task_ids,
            'constraint_names': self.constraint_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filepath: str):
        """Load dataset from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.features = data['features']
        self.labels = data['labels']
        self.task_ids = data['task_ids']
        self.constraint_names = data['constraint_names']

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        X, y = self.get_arrays()

        stats = {
            'n_samples': len(y),
            'n_positive': int(np.sum(y)),
            'n_negative': int(len(y) - np.sum(y)),
            'positive_rate': float(np.mean(y)),
            'n_tasks': len(set(self.task_ids)),
            'constraint_correlations': {}
        }

        # Compute correlation between each constraint and correctness
        for i, name in enumerate(self.constraint_names):
            constraint_values = X[:, i]
            # Pearson correlation
            correlation = np.corrcoef(constraint_values, y)[0, 1] if len(y) > 1 else 0.0
            stats['constraint_correlations'][name] = float(correlation)

        return stats


class ConstraintWeightLearner:
    """
    Learns optimal constraint weights from validation data.
    """

    def __init__(self, method='logistic'):
        """
        Initialize weight learner.

        Args:
            method: Learning method - 'logistic', 'grid_search', or 'gradient'
        """
        self.method = method
        self.weights = None
        self.constraint_names = [
            'color_preservation',
            'non_trivial',
            'size_consistency',
            'background_consistency',
            'color_count',
            'symmetry_preservation'
        ]

    def learn_logistic(self, X: np.ndarray, y: np.ndarray,
                      C=1.0, max_iter=1000) -> np.ndarray:
        """
        Learn weights using logistic regression.

        Args:
            X: Feature matrix (n_samples, n_constraints)
            y: Labels (n_samples,)
            C: Regularization strength (higher = less regularization)
            max_iter: Maximum iterations

        Returns:
            Learned weights (n_constraints,)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for logistic regression method")

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit logistic regression
        model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
        model.fit(X_scaled, y)

        # Get coefficients (weights)
        weights = model.coef_[0]

        # Convert back to original scale
        weights = weights / scaler.scale_

        # Make weights positive (we want positive correlation)
        weights = np.abs(weights)

        self.model = model
        self.scaler = scaler

        return weights

    def learn_grid_search(self, X: np.ndarray, y: np.ndarray,
                         weight_range=[0.1, 0.5, 1.0, 1.5, 2.0, 3.0],
                         n_top_weights=3) -> np.ndarray:
        """
        Learn weights using grid search (exhaustive but slow).

        For efficiency, only searches over top-N most correlated constraints.

        Args:
            X: Feature matrix
            y: Labels
            weight_range: Range of weights to try
            n_top_weights: Only tune top N constraints, set others to 1.0

        Returns:
            Learned weights
        """
        # Find top correlated constraints
        correlations = []
        for i in range(X.shape[1]):
            corr = np.corrcoef(X[:, i], y)[0, 1] if len(y) > 1 else 0.0
            correlations.append(abs(corr))

        top_indices = np.argsort(correlations)[-n_top_weights:]

        print(f"Grid search over top {n_top_weights} constraints:")
        for idx in top_indices:
            print(f"  - {self.constraint_names[idx]}: correlation = {correlations[idx]:.3f}")

        # Grid search
        best_weights = np.ones(X.shape[1])
        best_score = -np.inf

        import itertools
        for weight_combo in itertools.product(weight_range, repeat=n_top_weights):
            # Create weight vector
            weights = np.ones(X.shape[1])
            for i, idx in enumerate(top_indices):
                weights[idx] = weight_combo[i]

            # Evaluate
            score = self._evaluate_weights(X, y, weights)

            if score > best_score:
                best_score = score
                best_weights = weights.copy()

        print(f"Best score: {best_score:.4f}")
        return best_weights

    def learn_gradient(self, X: np.ndarray, y: np.ndarray,
                      lr=0.01, n_epochs=500, l2_reg=0.01) -> np.ndarray:
        """
        Learn weights using gradient descent.

        Args:
            X: Feature matrix
            y: Labels
            lr: Learning rate
            n_epochs: Number of training epochs
            l2_reg: L2 regularization strength

        Returns:
            Learned weights
        """
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        # Initialize weights to fixed constraint confidences
        initial_weights = torch.FloatTensor([0.87, 0.70, 0.80, 0.60, 0.60, 0.40])
        weights = torch.nn.Parameter(initial_weights)

        optimizer = torch.optim.Adam([weights], lr=lr)

        best_weights = weights.clone()
        best_loss = float('inf')

        for epoch in range(n_epochs):
            optimizer.zero_grad()

            # Compute scores: weighted sum of constraint features
            scores = torch.matmul(X_tensor, torch.abs(weights))  # abs to keep positive

            # Binary cross-entropy loss
            # Using sigmoid to convert scores to probabilities
            probs = torch.sigmoid(scores)
            loss = torch.nn.functional.binary_cross_entropy(probs, y_tensor)

            # Add L2 regularization
            loss = loss + l2_reg * torch.sum(weights ** 2)

            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_weights = weights.clone()

            if epoch % 100 == 0:
                accuracy = self._compute_accuracy(X, y, torch.abs(weights).detach().numpy())
                print(f"Epoch {epoch}: loss = {loss.item():.4f}, accuracy = {accuracy:.4f}")

        final_weights = torch.abs(best_weights).detach().numpy()
        return final_weights

    def _evaluate_weights(self, X: np.ndarray, y: np.ndarray,
                         weights: np.ndarray) -> float:
        """
        Evaluate weights using ranking accuracy.

        Score = how well the weights rank correct solutions higher than incorrect ones.
        """
        # Compute scores for all samples
        scores = np.dot(X, weights)

        # For each correct sample, count how many incorrect samples it beats
        correct_indices = np.where(y == 1)[0]
        incorrect_indices = np.where(y == 0)[0]

        if len(correct_indices) == 0 or len(incorrect_indices) == 0:
            return 0.0

        # Pairwise comparison
        wins = 0
        total = 0
        for c_idx in correct_indices:
            for i_idx in incorrect_indices:
                if scores[c_idx] > scores[i_idx]:
                    wins += 1
                total += 1

        return wins / total if total > 0 else 0.0

    def _compute_accuracy(self, X: np.ndarray, y: np.ndarray,
                         weights: np.ndarray) -> float:
        """Compute classification accuracy."""
        scores = np.dot(X, weights)
        predictions = (scores > np.median(scores)).astype(float)
        accuracy = np.mean(predictions == y)
        return accuracy

    def fit(self, dataset: MetaLearningDataset, **kwargs) -> Dict[str, float]:
        """
        Fit weights to dataset.

        Args:
            dataset: MetaLearningDataset instance
            **kwargs: Method-specific arguments

        Returns:
            Dictionary mapping constraint names to learned weights
        """
        X, y = dataset.get_arrays()

        print(f"\nLearning constraint weights using method: {self.method}")
        print(f"Dataset: {len(y)} samples, {np.sum(y):.0f} correct, {len(y)-np.sum(y):.0f} incorrect")

        if self.method == 'logistic':
            weights_array = self.learn_logistic(X, y, **kwargs)
        elif self.method == 'grid_search':
            weights_array = self.learn_grid_search(X, y, **kwargs)
        elif self.method == 'gradient':
            weights_array = self.learn_gradient(X, y, **kwargs)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Convert to dictionary
        self.weights = {name: float(weight)
                       for name, weight in zip(self.constraint_names, weights_array)}

        print("\nLearned weights:")
        for name, weight in self.weights.items():
            print(f"  {name}: {weight:.3f}")

        return self.weights

    def save_weights(self, filepath: str):
        """Save learned weights to JSON file."""
        if self.weights is None:
            raise ValueError("No weights learned yet. Call fit() first.")

        with open(filepath, 'w') as f:
            json.dump(self.weights, f, indent=2)

        print(f"Weights saved to {filepath}")

    def load_weights(self, filepath: str) -> Dict[str, float]:
        """Load weights from JSON file."""
        with open(filepath, 'r') as f:
            self.weights = json.load(f)

        print(f"Weights loaded from {filepath}")
        return self.weights


def collect_meta_learning_data(tasks, loggers, solutions_file: str) -> MetaLearningDataset:
    """
    Collect meta-learning dataset from validation tasks.

    Args:
        tasks: List of preprocessing.Task objects
        loggers: List of EnhancedLogger objects with tracked candidates
        solutions_file: Path to ground truth solutions JSON

    Returns:
        MetaLearningDataset with constraint features and labels
    """
    # Load ground truth
    with open(solutions_file, 'r') as f:
        solutions = json.load(f)

    dataset = MetaLearningDataset()

    for task, logger in zip(tasks, loggers):
        task_name = task.task_name
        if task_name not in solutions:
            continue

        # Create constraint validator for this task
        validator = arc_constraints.ARCConstraintValidator(task)
        extractor = ConstraintFeatureExtractor(validator)

        # Get all tracked candidates
        for hash_key, candidate_info in logger.candidate_solutions.items():
            candidate_solution = candidate_info['solution']

            if candidate_solution is None:
                continue

            # For each test example in this candidate
            for test_idx in range(task.n_test):
                if test_idx >= len(candidate_solution):
                    continue

                # Get input grid
                input_grid = task.problem[task.n_train + test_idx, :, :, 0].cpu().numpy()
                input_shape = task.shapes[task.n_train + test_idx][0]
                input_cropped = input_grid[:input_shape[0], :input_shape[1]]

                # Get predicted output
                predicted_output = candidate_solution[test_idx]

                # Check if correct
                ground_truth = solutions[task_name][test_idx]
                is_correct = (predicted_output == tuple(tuple(row) for row in ground_truth))

                # Extract features
                features = extractor.extract_features(input_cropped, predicted_output)

                # Add to dataset
                dataset.add_sample(features, is_correct, task_name)

    print(f"\nCollected meta-learning dataset:")
    stats = dataset.get_statistics()
    print(f"  Total samples: {stats['n_samples']}")
    print(f"  Correct: {stats['n_positive']}")
    print(f"  Incorrect: {stats['n_negative']}")
    print(f"  Positive rate: {stats['positive_rate']:.3f}")
    print(f"  Tasks: {stats['n_tasks']}")
    print("\nConstraint correlations with correctness:")
    for name, corr in stats['constraint_correlations'].items():
        print(f"  {name}: {corr:.3f}")

    return dataset
