"""
ARC-AGI Constraint Validators

This module implements constraint validators based on common patterns in ARC-AGI tasks.
These constraints help select better solutions by filtering/scoring candidates based on
structural regularities observed in ARC tasks.

Key insight: Most ARC tasks follow certain rules (e.g., output colors are subset of input colors,
symmetries are preserved, object counts remain stable). We can use these as soft or hard constraints.
"""

import numpy as np
from typing import List, Tuple, Dict, Set
from collections import Counter
import torch


class ARCConstraintValidator:
    """
    Validates and scores solutions based on ARC task constraints.
    """

    def __init__(self, task):
        """
        Initialize validator with training examples to infer constraints.

        Args:
            task: preprocessing.Task object containing training examples
        """
        self.task = task
        self.train_input_grids = []
        self.train_output_grids = []

        # Extract training grids
        for example_num in range(task.n_train):
            input_grid = task.problem[example_num, :, :, 0]
            output_grid = task.problem[example_num, :, :, 1]
            input_shape = task.shapes[example_num][0]
            output_shape = task.shapes[example_num][1]

            # Crop to actual size
            input_cropped = input_grid[:input_shape[0], :input_shape[1]].cpu().numpy()
            output_cropped = output_grid[:output_shape[0], :output_shape[1]].cpu().numpy()

            self.train_input_grids.append(input_cropped)
            self.train_output_grids.append(output_cropped)

        # Infer meta-constraints from training examples
        self.meta_constraints = self._infer_meta_constraints()

    def _infer_meta_constraints(self) -> Dict:
        """
        Analyze training examples to infer task-level constraints.

        Returns:
            Dictionary of meta-constraints learned from training data
        """
        meta = {
            'color_preservation_rate': 0.0,
            'color_reduction': True,
            'size_relationship': None,
            'output_colors_subset': True,
            'same_background': True,
            'all_outputs_same_size': self.task.all_out_same_size,
            'in_out_same_size': self.task.in_out_same_size,
        }

        # Check color preservation across training examples
        color_preserved_count = 0
        for inp, out in zip(self.train_input_grids, self.train_output_grids):
            input_colors = set(np.unique(inp))
            output_colors = set(np.unique(out))

            if output_colors.issubset(input_colors):
                color_preserved_count += 1

        meta['color_preservation_rate'] = color_preserved_count / len(self.train_input_grids)
        meta['color_reduction'] = all(
            len(np.unique(out)) <= len(np.unique(inp))
            for inp, out in zip(self.train_input_grids, self.train_output_grids)
        )

        # Check size relationships
        size_ratios = []
        for inp, out in zip(self.train_input_grids, self.train_output_grids):
            ratio_h = out.shape[0] / inp.shape[0] if inp.shape[0] > 0 else 1
            ratio_w = out.shape[1] / inp.shape[1] if inp.shape[1] > 0 else 1
            size_ratios.append((ratio_h, ratio_w))

        # Check if size ratios are consistent
        if len(set(size_ratios)) == 1:
            meta['size_relationship'] = size_ratios[0]

        return meta

    def validate_color_preservation(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Tuple[bool, float]:
        """
        Validate that output colors are a subset of input colors.

        Returns:
            (is_valid, confidence_score)
        """
        input_colors = set(np.unique(input_grid))
        output_colors = set(np.unique(output_grid))

        is_valid = output_colors.issubset(input_colors)
        confidence = self.meta_constraints['color_preservation_rate']

        return is_valid, confidence

    def validate_non_trivial(self, output_grid: np.ndarray) -> Tuple[bool, float]:
        """
        Validate that output is not trivially constant (all same color).

        Returns:
            (is_valid, confidence_score)
        """
        unique_colors = len(np.unique(output_grid))

        # Most ARC tasks have non-trivial outputs
        is_valid = unique_colors > 1 or output_grid.size <= 1
        confidence = 0.7  # This is a weak constraint

        return is_valid, confidence

    def validate_size_consistency(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Tuple[bool, float]:
        """
        Validate size relationship matches training examples.

        Returns:
            (is_valid, confidence_score)
        """
        if self.meta_constraints['size_relationship'] is None:
            return True, 0.1  # No clear pattern

        expected_ratio = self.meta_constraints['size_relationship']
        actual_ratio = (
            output_grid.shape[0] / input_grid.shape[0] if input_grid.shape[0] > 0 else 1,
            output_grid.shape[1] / input_grid.shape[1] if input_grid.shape[1] > 0 else 1
        )

        is_valid = abs(expected_ratio[0] - actual_ratio[0]) < 0.01 and \
                   abs(expected_ratio[1] - actual_ratio[1]) < 0.01

        confidence = 0.8 if is_valid else 0.5

        return is_valid, confidence

    def validate_background_consistency(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Tuple[bool, float]:
        """
        Validate that background color (most common) is handled consistently.

        Returns:
            (is_valid, confidence_score)
        """
        # Get most common color (background) from input
        input_bg = Counter(input_grid.flatten()).most_common(1)[0][0]
        output_colors = set(np.unique(output_grid))

        # Check if background is preserved or removed
        train_bg_preserved = []
        for inp, out in zip(self.train_input_grids, self.train_output_grids):
            inp_bg = Counter(inp.flatten()).most_common(1)[0][0]
            train_bg_preserved.append(inp_bg in np.unique(out))

        if len(train_bg_preserved) > 0:
            bg_preservation_rate = sum(train_bg_preserved) / len(train_bg_preserved)

            if bg_preservation_rate > 0.7:
                # Background usually preserved
                is_valid = input_bg in output_colors
                confidence = bg_preservation_rate
            elif bg_preservation_rate < 0.3:
                # Background usually removed
                is_valid = input_bg not in output_colors
                confidence = 1 - bg_preservation_rate
            else:
                # Unclear pattern
                is_valid = True
                confidence = 0.1
        else:
            is_valid = True
            confidence = 0.1

        return is_valid, confidence

    def validate_color_count(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Tuple[bool, float]:
        """
        Validate number of unique colors based on training pattern.

        Returns:
            (is_valid, confidence_score)
        """
        input_color_count = len(np.unique(input_grid))
        output_color_count = len(np.unique(output_grid))

        if self.meta_constraints['color_reduction']:
            is_valid = output_color_count <= input_color_count
            confidence = 0.6
        else:
            is_valid = True
            confidence = 0.1

        return is_valid, confidence

    def validate_symmetry_preservation(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Tuple[bool, float]:
        """
        If input has symmetry, output often should too.

        Returns:
            (is_valid, confidence_score)
        """
        def has_vertical_symmetry(grid):
            return np.array_equal(grid, np.fliplr(grid))

        def has_horizontal_symmetry(grid):
            return np.array_equal(grid, np.flipud(grid))

        input_v_sym = has_vertical_symmetry(input_grid)
        input_h_sym = has_horizontal_symmetry(input_grid)

        output_v_sym = has_vertical_symmetry(output_grid)
        output_h_sym = has_horizontal_symmetry(output_grid)

        # If input has symmetry, prefer outputs with symmetry
        if input_v_sym or input_h_sym:
            is_valid = output_v_sym or output_h_sym
            confidence = 0.4  # Weak constraint
        else:
            is_valid = True
            confidence = 0.1

        return is_valid, confidence

    def score_solution(self, test_input_grid: np.ndarray, candidate_output: List[List[int]],
                      base_uncertainty: float = 0.0) -> float:
        """
        Score a candidate solution using all constraints.

        Args:
            test_input_grid: Input grid for this test example
            candidate_output: Predicted output grid (list of lists)
            base_uncertainty: Uncertainty from model predictions

        Returns:
            Combined score (higher is better)
        """
        output_array = np.array(candidate_output, dtype=int)

        # Map colors back to original indices
        color_mapping = {self.task.colors[i]: i for i in range(len(self.task.colors))}
        input_remapped = np.array([[color_mapping.get(int(val), int(val)) for val in row]
                                   for row in test_input_grid], dtype=int)
        output_remapped = np.array([[color_mapping.get(int(val), int(val)) for val in row]
                                     for row in output_array], dtype=int)

        # Apply all constraints
        constraints = [
            self.validate_color_preservation(input_remapped, output_remapped),
            self.validate_non_trivial(output_remapped),
            self.validate_size_consistency(input_remapped, output_remapped),
            self.validate_background_consistency(input_remapped, output_remapped),
            self.validate_color_count(input_remapped, output_remapped),
            self.validate_symmetry_preservation(input_remapped, output_remapped),
        ]

        # Compute weighted score
        total_score = 0.0
        total_weight = 0.0

        for is_valid, confidence in constraints:
            weight = confidence
            score = 1.0 if is_valid else 0.0
            total_score += score * weight
            total_weight += weight

        # Normalize and combine with uncertainty
        if total_weight > 0:
            constraint_score = total_score / total_weight
        else:
            constraint_score = 0.5

        # Combine: constraint score (0-1) + uncertainty penalty
        final_score = constraint_score * 10 - base_uncertainty

        return final_score


def compute_grid_similarity(grid1: np.ndarray, grid2: np.ndarray) -> float:
    """
    Compute similarity between two grids (0-1, higher is more similar).
    """
    if grid1.shape != grid2.shape:
        return 0.0

    matching_pixels = np.sum(grid1 == grid2)
    total_pixels = grid1.size

    return matching_pixels / total_pixels if total_pixels > 0 else 0.0


def get_connected_components(grid: np.ndarray, background: int = 0) -> int:
    """
    Count number of connected components (objects) in grid.
    """
    from scipy import ndimage

    # Create binary mask (non-background)
    mask = grid != background

    # Label connected components
    labeled, num_components = ndimage.label(mask)

    return num_components
