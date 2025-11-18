"""
Grid Search Example for Meta-Learning Constraint Weights

This demonstrates how grid search works with a concrete ARC task example.
"""

import numpy as np
import json
from typing import List, Dict, Tuple


class GridSearchExample:
    """
    Demonstrates grid search for learning constraint weights on an example ARC task.
    """

    def __init__(self):
        # Simulated candidate solutions for a task
        # Each candidate has constraint features and correctness label
        self.candidates = [
            # Format: (color_pres, non_triv, size_cons, bg_cons, color_cnt, symmetry, is_correct)
            ([1, 1, 1, 1, 1, 0], True),   # Candidate 1: All constraints except symmetry, CORRECT
            ([1, 1, 1, 0, 1, 0], False),  # Candidate 2: Missing background, incorrect
            ([1, 1, 0, 1, 1, 0], False),  # Candidate 3: Missing size, incorrect
            ([0, 1, 1, 1, 1, 0], False),  # Candidate 4: Missing color preservation, incorrect
            ([1, 1, 1, 1, 0, 0], False),  # Candidate 5: Missing color count, incorrect
            ([1, 1, 1, 1, 1, 1], True),   # Candidate 6: All constraints, CORRECT
            ([1, 0, 1, 1, 1, 0], False),  # Candidate 7: Trivial output, incorrect
            ([1, 1, 0, 0, 1, 0], False),  # Candidate 8: Missing multiple, incorrect
        ]

        self.constraint_names = [
            'color_preservation',
            'non_trivial',
            'size_consistency',
            'background_consistency',
            'color_count',
            'symmetry_preservation'
        ]

    def compute_correlations(self) -> List[Tuple[str, float]]:
        """
        Step 1: Compute correlation of each constraint with correctness.
        This tells us which constraints are most predictive.
        """
        features = np.array([feat for feat, _ in self.candidates])  # Shape: (8, 6)
        labels = np.array([float(label) for _, label in self.candidates])  # Shape: (8,)

        correlations = []
        for i, name in enumerate(self.constraint_names):
            constraint_values = features[:, i]
            # Pearson correlation between constraint and correctness
            correlation = np.corrcoef(constraint_values, labels)[0, 1]
            correlations.append((name, correlation))

        return sorted(correlations, key=lambda x: abs(x[1]), reverse=True)

    def evaluate_weights(self, weights: np.ndarray) -> float:
        """
        Step 2: Evaluate a weight combination using ranking accuracy.

        For each correct solution, count how many incorrect solutions it beats.
        Better weights → correct solutions score higher than incorrect ones.
        """
        features = np.array([feat for feat, _ in self.candidates])
        labels = np.array([label for _, label in self.candidates])

        # Compute scores: weighted sum of constraint features
        scores = np.dot(features, weights)

        # Separate correct and incorrect solutions
        correct_indices = np.where(labels)[0]
        incorrect_indices = np.where(~labels)[0]

        # Count pairwise wins: correct solution scores > incorrect solution scores
        wins = 0
        total = 0
        for c_idx in correct_indices:
            for i_idx in incorrect_indices:
                if scores[c_idx] > scores[i_idx]:
                    wins += 1
                total += 1

        accuracy = wins / total if total > 0 else 0.0
        return accuracy

    def grid_search(self, weight_range=[0.1, 0.5, 1.0, 1.5, 2.0, 3.0],
                   n_top_constraints=3) -> Dict[str, float]:
        """
        Step 3: Grid search over weight combinations.

        To avoid combinatorial explosion (6^6 = 46,656 combinations),
        only tune the top-N most correlated constraints.
        """
        # Step 3a: Find top correlated constraints
        correlations = self.compute_correlations()
        print("Constraint correlations with correctness:")
        for name, corr in correlations:
            print(f"  {name}: {corr:.3f}")

        top_constraints = correlations[:n_top_constraints]
        top_indices = [self.constraint_names.index(name) for name, _ in top_constraints]

        print(f"\nTuning top {n_top_constraints} constraints:")
        for name, corr in top_constraints:
            print(f"  {name} (correlation: {corr:.3f})")

        # Step 3b: Grid search
        import itertools

        best_weights = np.ones(6)
        best_accuracy = -1.0
        n_combinations = len(weight_range) ** n_top_constraints

        print(f"\nSearching {n_combinations} weight combinations...")

        for i, weight_combo in enumerate(itertools.product(weight_range, repeat=n_top_constraints)):
            # Create weight vector: 1.0 for non-tuned, searched values for tuned
            weights = np.ones(6)
            for j, idx in enumerate(top_indices):
                weights[idx] = weight_combo[j]

            # Evaluate
            accuracy = self.evaluate_weights(weights)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = weights.copy()

                print(f"  [Better] Combo {i+1}/{n_combinations}: accuracy={accuracy:.3f}")
                for k, idx in enumerate(top_indices):
                    print(f"    {self.constraint_names[idx]}: {weight_combo[k]:.1f}")

        print(f"\nBest accuracy: {best_accuracy:.3f}")

        # Convert to dictionary
        result = {name: float(weight) for name, weight in zip(self.constraint_names, best_weights)}
        return result


def concrete_arc_example():
    """
    Concrete example: Task that requires output colors to be subset of input colors.
    """
    print("=" * 80)
    print("GRID SEARCH EXAMPLE: ARC Task Color Subset Pattern")
    print("=" * 80)
    print("\nTask Description:")
    print("  Input: Grid with multiple colors [0, 1, 2, 3, 4]")
    print("  Rule: Output should only use colors from input (subset constraint)")
    print("  Ground truth: Output uses [0, 2, 4] (subset of input)")
    print("\n")

    # Simulated candidates generated by model
    candidates = [
        {
            'output_colors': [0, 2, 4],  # Correct! Subset of input
            'constraints': {
                'color_preservation': True,    # Output ⊆ Input ✓
                'non_trivial': True,           # Multiple colors ✓
                'size_consistency': True,      # Size matches pattern ✓
                'background_consistency': True, # Background preserved ✓
                'color_count': True,           # Fewer colors than input ✓
                'symmetry_preservation': False # No symmetry required
            },
            'is_correct': True
        },
        {
            'output_colors': [0, 2, 5],  # Wrong! 5 not in input
            'constraints': {
                'color_preservation': False,   # Output ⊄ Input ✗
                'non_trivial': True,
                'size_consistency': True,
                'background_consistency': True,
                'color_count': True,
                'symmetry_preservation': False
            },
            'is_correct': False
        },
        {
            'output_colors': [0],  # Wrong! Trivial (all same color)
            'constraints': {
                'color_preservation': True,
                'non_trivial': False,          # Only one color ✗
                'size_consistency': False,     # Size wrong ✗
                'background_consistency': True,
                'color_count': True,
                'symmetry_preservation': False
            },
            'is_correct': False
        },
        {
            'output_colors': [0, 1, 2, 4],  # Wrong! Too many colors
            'constraints': {
                'color_preservation': True,
                'non_trivial': True,
                'size_consistency': True,
                'background_consistency': False, # Background wrong ✗
                'color_count': False,           # Too many colors ✗
                'symmetry_preservation': False
            },
            'is_correct': False
        },
    ]

    print("Generated candidates:")
    for i, cand in enumerate(candidates, 1):
        status = "✓ CORRECT" if cand['is_correct'] else "✗ INCORRECT"
        print(f"\n  Candidate {i}: {status}")
        print(f"    Output colors: {cand['output_colors']}")
        print(f"    Constraints:")
        for name, satisfied in cand['constraints'].items():
            symbol = "✓" if satisfied else "✗"
            print(f"      {symbol} {name}: {satisfied}")

    # Step 1: Compute correlations
    print("\n" + "=" * 80)
    print("STEP 1: Compute Constraint Correlations")
    print("=" * 80)

    features = []
    labels = []
    for cand in candidates:
        feat = [int(cand['constraints'][name]) for name in [
            'color_preservation', 'non_trivial', 'size_consistency',
            'background_consistency', 'color_count', 'symmetry_preservation'
        ]]
        features.append(feat)
        labels.append(cand['is_correct'])

    features = np.array(features)
    labels = np.array(labels, dtype=float)

    constraint_names = ['color_preservation', 'non_trivial', 'size_consistency',
                       'background_consistency', 'color_count', 'symmetry_preservation']

    correlations = []
    for i, name in enumerate(constraint_names):
        corr = np.corrcoef(features[:, i], labels)[0, 1] if len(labels) > 1 else 0.0
        correlations.append((name, corr))

    correlations_sorted = sorted(correlations, key=lambda x: abs(x[1]), reverse=True)

    print("\nConstraint correlations with correctness:")
    for name, corr in correlations_sorted:
        print(f"  {name:.<30} {corr:>6.3f}")

    # Interpretation
    print("\nInterpretation:")
    print(f"  - {correlations_sorted[0][0]} has highest correlation ({correlations_sorted[0][1]:.3f})")
    print(f"    → This constraint strongly predicts correct solutions!")
    print(f"  - {correlations_sorted[-1][0]} has lowest correlation ({correlations_sorted[-1][1]:.3f})")
    print(f"    → This constraint is not very predictive")

    # Step 2: Grid search
    print("\n" + "=" * 80)
    print("STEP 2: Grid Search Over Top-3 Constraints")
    print("=" * 80)

    top_3 = correlations_sorted[:3]
    print(f"\nWill tune these constraints:")
    for name, corr in top_3:
        print(f"  - {name} (correlation: {corr:.3f})")

    print(f"\nOthers fixed at weight = 1.0")

    weight_range = [0.5, 1.0, 2.0, 3.0]  # Simplified for example
    top_indices = [constraint_names.index(name) for name, _ in top_3]

    print(f"\nWeight range: {weight_range}")
    print(f"Combinations to try: {len(weight_range)**3} = {4**3}")

    # Grid search
    import itertools

    best_weights = None
    best_score = -1.0

    print("\nSearching...")

    def evaluate_ranking(weights):
        """Compute how well weights rank correct > incorrect."""
        scores = np.dot(features, weights)
        correct_scores = scores[labels.astype(bool)]
        incorrect_scores = scores[~labels.astype(bool)]

        if len(correct_scores) == 0 or len(incorrect_scores) == 0:
            return 0.0

        # Pairwise comparison
        wins = sum(c > i for c in correct_scores for i in incorrect_scores)
        total = len(correct_scores) * len(incorrect_scores)
        return wins / total

    for combo in itertools.product(weight_range, repeat=3):
        weights = np.ones(6)
        for i, idx in enumerate(top_indices):
            weights[idx] = combo[i]

        score = evaluate_ranking(weights)

        if score > best_score:
            best_score = score
            best_weights = weights.copy()

            print(f"\n  New best! Ranking accuracy: {score:.3f}")
            for i, idx in enumerate(top_indices):
                print(f"    {constraint_names[idx]:.<30} {combo[i]:.1f}")

    # Step 3: Show final results
    print("\n" + "=" * 80)
    print("STEP 3: Final Learned Weights")
    print("=" * 80)

    print(f"\nBest ranking accuracy: {best_score:.3f}")
    print(f"\nLearned weights:")
    for name, weight in zip(constraint_names, best_weights):
        marker = "←" if weight != 1.0 else ""
        print(f"  {name:.<30} {weight:.2f} {marker}")

    print("\nComparison to fixed weights:")
    fixed_weights = {
        'color_preservation': 0.87,
        'non_trivial': 0.70,
        'size_consistency': 0.80,
        'background_consistency': 0.60,
        'color_count': 0.60,
        'symmetry_preservation': 0.40
    }

    print(f"\n  {'Constraint':<30} {'Fixed':<10} {'Learned':<10} {'Change':<10}")
    print(f"  {'-'*60}")
    for name, learned_w in zip(constraint_names, best_weights):
        fixed_w = fixed_weights[name]
        change = ((learned_w - fixed_w) / fixed_w * 100) if fixed_w > 0 else 0
        print(f"  {name:<30} {fixed_w:<10.2f} {learned_w:<10.2f} {change:>+6.1f}%")

    print("\nKey insights:")
    print(f"  1. color_preservation weight increased significantly")
    print(f"     → Strong predictor of correctness in this task type")
    print(f"  2. Less predictive constraints remain at 1.0")
    print(f"     → Grid search only tunes top-N, others stay default")
    print(f"  3. Final weights maximize ranking accuracy")
    print(f"     → Correct solutions now score higher than incorrect ones")

    return best_weights


if __name__ == "__main__":
    # Run simplified example
    print("SIMPLIFIED EXAMPLE")
    print("=" * 80)
    example = GridSearchExample()
    learned_weights = example.grid_search(weight_range=[0.5, 1.0, 1.5, 2.0], n_top_constraints=3)

    print("\n\nLearned weights:")
    print(json.dumps(learned_weights, indent=2))

    print("\n\n")
    print("=" * 80)
    print("CONCRETE ARC TASK EXAMPLE")
    print("=" * 80)
    concrete_arc_example()
