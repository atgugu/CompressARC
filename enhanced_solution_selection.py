"""
Enhanced Solution Selection with Constraint-Based Validation

This module extends the basic solution_selection.py with:
1. Diverse candidate generation (multiple sampling strategies)
2. Constraint-based scoring using arc_constraints.py
3. Better solution ranking and selection

Key improvements:
- Generate 10-20+ candidates instead of just 2
- Use ARC structural constraints to score candidates
- Combine uncertainty with constraint validation
- More robust to model uncertainty
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Tuple, Dict
from collections import defaultdict

import arc_constraints


class EnhancedLogger:
    """
    Enhanced logger with diverse sampling and constraint-based validation.
    """
    ema_decay = 0.97

    def __init__(self, task, use_constraints=True, n_candidates=20, learned_weights=None):
        """
        Initialize enhanced logger.

        Args:
            task: preprocessing.Task object
            use_constraints: Whether to use constraint-based scoring
            n_candidates: Number of diverse candidates to generate
            learned_weights: Optional dict of learned constraint weights.
                           If provided, uses these instead of fixed weights.
        """
        self.task = task
        self.use_constraints = use_constraints
        self.n_candidates = n_candidates

        # Initialize constraint validator with learned weights
        if use_constraints:
            try:
                self.validator = arc_constraints.ARCConstraintValidator(task, learned_weights=learned_weights)
            except Exception as e:
                print(f"Warning: Could not initialize constraint validator: {e}")
                self.validator = None
                self.use_constraints = False
        else:
            self.validator = None

        # Logging structures
        self.KL_curves = {}
        self.total_KL_curve = []
        self.reconstruction_error_curve = []
        self.loss_curve = []

        n_test, n_colors, n_x, n_y = task.n_test, task.n_colors, task.n_x, task.n_y
        shape = (n_test, n_colors + 1, n_x, n_y)

        self.current_logits = torch.zeros(shape)
        self.current_x_mask = torch.zeros((n_test, n_x))
        self.current_y_mask = torch.zeros((n_test, n_y))

        self.ema_logits = torch.zeros(shape)
        self.ema_x_mask = torch.zeros((n_test, n_x))
        self.ema_y_mask = torch.zeros((n_test, n_y))

        # Enhanced: Track more solutions
        self.candidate_solutions = defaultdict(lambda: {'solution': None, 'score': -np.inf, 'count': 0})
        self.solution_most_frequent = None
        self.solution_second_most_frequent = None

        self.solution_contributions_log = []
        self.solution_picks_history = []

    def log(self, train_step, logits, x_mask, y_mask, KL_amounts, KL_names, total_KL, reconstruction_error, loss):
        """Logs training progress and tracks solutions from one forward pass."""
        if train_step == 0:
            self.KL_curves = {KL_name: [] for KL_name in KL_names}

        for KL_amount, KL_name in zip(KL_amounts, KL_names):
            self.KL_curves[KL_name].append(float(KL_amount.detach().sum().cpu().numpy()))

        self.total_KL_curve.append(float(total_KL.detach().cpu().numpy()))
        self.reconstruction_error_curve.append(float(reconstruction_error.detach().cpu().numpy()))
        self.loss_curve.append(float(loss.detach().cpu().numpy()))

        self._track_solutions_enhanced(train_step, logits.detach(), x_mask.detach(), y_mask.detach())

    def _track_solutions_enhanced(self, train_step, logits, x_mask, y_mask):
        """
        Enhanced solution tracking with diverse sampling and constraint-based scoring.
        """
        self.current_logits = logits[self.task.n_train:, :, :, :, 1]
        self.current_x_mask = x_mask[self.task.n_train:, :, 1]
        self.current_y_mask = y_mask[self.task.n_train:, :, 1]

        self.ema_logits = self.ema_decay * self.ema_logits + (1 - self.ema_decay) * self.current_logits
        self.ema_x_mask = self.ema_decay * self.ema_x_mask + (1 - self.ema_decay) * self.current_x_mask
        self.ema_y_mask = self.ema_decay * self.ema_y_mask + (1 - self.ema_decay) * self.current_y_mask

        # Generate diverse candidates
        candidates = self._generate_diverse_candidates(train_step)

        solution_contributions = []
        for candidate_info in candidates:
            solution = candidate_info['solution']
            uncertainty = candidate_info['uncertainty']
            sampling_method = candidate_info['method']

            hashed_solution = hash(solution)

            # Base score from uncertainty
            score = -10 * uncertainty

            # Time-based penalty for early steps
            if train_step < 150:
                score = score - 10

            # Method-specific penalties
            if sampling_method == 'ema':
                score = score - 4
            elif 'sample' in sampling_method:
                score = score - 2  # Slight penalty for stochastic methods

            # Enhanced: Add constraint-based scoring
            if self.use_constraints and self.validator is not None:
                constraint_score = self._score_with_constraints(solution, uncertainty)
                score = score + constraint_score

            # Track this candidate
            solution_contributions.append((hashed_solution, score))

            # Update candidate tracking
            prev_score = self.candidate_solutions[hashed_solution]['score']
            # Use log-sum-exp for score accumulation
            self.candidate_solutions[hashed_solution]['score'] = float(
                np.logaddexp(prev_score, score))
            self.candidate_solutions[hashed_solution]['solution'] = solution
            self.candidate_solutions[hashed_solution]['count'] += 1

        # Update top solutions
        self._update_top_solutions()

        self.solution_contributions_log.append(solution_contributions)
        self.solution_picks_history.append([
            hash(sol) for sol in [self.solution_most_frequent, self.solution_second_most_frequent]
        ])

    def _generate_diverse_candidates(self, train_step) -> List[Dict]:
        """
        Generate diverse candidate solutions using multiple sampling strategies.

        Returns:
            List of candidate dictionaries with keys: 'solution', 'uncertainty', 'method'
        """
        candidates = []

        # Strategy 1: Current argmax (greedy)
        solution, uncertainty = self._postprocess_solution(
            self.current_logits, self.current_x_mask, self.current_y_mask, use_argmax=True)
        candidates.append({'solution': solution, 'uncertainty': uncertainty, 'method': 'argmax_current'})

        # Strategy 2: EMA argmax (stable estimate)
        solution, uncertainty = self._postprocess_solution(
            self.ema_logits, self.ema_x_mask, self.ema_y_mask, use_argmax=True)
        candidates.append({'solution': solution, 'uncertainty': uncertainty, 'method': 'ema'})

        # Strategy 3-5: Temperature-based sampling (diverse exploration)
        for temp_idx, temperature in enumerate([0.5, 1.0, 1.5]):
            solution, uncertainty = self._postprocess_solution(
                self.current_logits, self.current_x_mask, self.current_y_mask,
                use_argmax=False, temperature=temperature)
            candidates.append({
                'solution': solution,
                'uncertainty': uncertainty,
                'method': f'sample_temp{temperature}'
            })

        # Strategy 6-7: Nucleus (top-p) sampling
        for p in [0.9, 0.95]:
            solution, uncertainty = self._postprocess_solution(
                self.current_logits, self.current_x_mask, self.current_y_mask,
                use_argmax=False, nucleus_p=p)
            candidates.append({
                'solution': solution,
                'uncertainty': uncertainty,
                'method': f'nucleus_p{p}'
            })

        # Strategy 8-10: Hybrid EMA + sampling (for later in training)
        if train_step > 500:
            for temp in [0.7, 1.0]:
                solution, uncertainty = self._postprocess_solution(
                    self.ema_logits, self.ema_x_mask, self.ema_y_mask,
                    use_argmax=False, temperature=temp)
                candidates.append({
                    'solution': solution,
                    'uncertainty': uncertainty,
                    'method': f'ema_sample_temp{temp}'
                })

        # Limit to n_candidates
        return candidates[:self.n_candidates]

    def _score_with_constraints(self, solution, base_uncertainty: float) -> float:
        """
        Score solution using ARC constraints.

        Returns:
            Additional score based on constraint satisfaction
        """
        if self.validator is None:
            return 0.0

        total_constraint_score = 0.0

        # Score each test example
        for test_idx in range(self.task.n_test):
            # Get input grid for this test example
            input_grid = self.task.problem[self.task.n_train + test_idx, :, :, 0].cpu().numpy()
            input_shape = self.task.shapes[self.task.n_train + test_idx][0]
            input_cropped = input_grid[:input_shape[0], :input_shape[1]]

            # Get predicted output
            predicted_output = solution[test_idx]

            # Score this example
            example_score = self.validator.score_solution(
                input_cropped, predicted_output, base_uncertainty)

            total_constraint_score += example_score

        # Average across test examples
        return total_constraint_score / self.task.n_test if self.task.n_test > 0 else 0.0

    def _update_top_solutions(self):
        """Update the top two solutions based on accumulated scores."""
        if not self.candidate_solutions:
            return

        # Sort by score
        sorted_candidates = sorted(
            self.candidate_solutions.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )

        if len(sorted_candidates) >= 1:
            self.solution_most_frequent = sorted_candidates[0][1]['solution']
        if len(sorted_candidates) >= 2:
            self.solution_second_most_frequent = sorted_candidates[1][1]['solution']

    def best_crop(self, prediction, x_mask, x_length, y_mask, y_length):
        """Crop prediction to best-scoring region."""
        x_start, x_end = self._best_slice_point(x_mask, x_length)
        y_start, y_end = self._best_slice_point(y_mask, y_length)
        return prediction[..., x_start:x_end, y_start:y_end]

    def _best_slice_point(self, mask, length):
        """Find best slice point for cropping."""
        if self.task.in_out_same_size or self.task.all_out_same_size:
            search_lengths = [length]
        else:
            search_lengths = list(range(1, mask.shape[0] + 1))
        max_logprob, best_slice_start, best_slice_end = None, None, None

        for length in search_lengths:
            logprobs = torch.stack([
                -torch.sum(mask[:offset]) + torch.sum(mask[offset:offset + length]) - torch.sum(mask[offset + length:])
                for offset in range(mask.shape[0] - length + 1)
            ])
            if max_logprob is None or torch.max(logprobs) > max_logprob:
                max_logprob = torch.max(logprobs)
                best_slice_start = torch.argmax(logprobs).item()
                best_slice_end = best_slice_start + length

        return best_slice_start, best_slice_end

    def _postprocess_solution(self, prediction, x_mask, y_mask, use_argmax=True,
                             temperature=1.0, nucleus_p=None):
        """
        Postprocess solution with optional sampling strategies.

        Args:
            prediction: Logits tensor (example, color, x, y)
            x_mask: X dimension mask
            y_mask: Y dimension mask
            use_argmax: If True, take argmax; else sample
            temperature: Temperature for sampling
            nucleus_p: If provided, use nucleus sampling with this threshold

        Returns:
            (solution, uncertainty)
        """
        if use_argmax:
            colors = torch.argmax(prediction, dim=1)  # example, x, y
        else:
            # Sample from distribution
            if nucleus_p is not None:
                colors = self._nucleus_sample(prediction, nucleus_p)
            else:
                probs = torch.softmax(prediction / temperature, dim=1)
                # Sample for each pixel
                colors = torch.zeros(prediction.shape[0], prediction.shape[2], prediction.shape[3], dtype=torch.long)
                for ex in range(prediction.shape[0]):
                    for x in range(prediction.shape[2]):
                        for y in range(prediction.shape[3]):
                            pixel_probs = probs[ex, :, x, y].cpu().numpy()
                            colors[ex, x, y] = np.random.choice(len(pixel_probs), p=pixel_probs)

        uncertainties = torch.logsumexp(prediction, dim=1) - torch.amax(prediction, dim=1)
        solution_slices, uncertainty_values = [], []

        for example_num in range(self.task.n_test):
            x_length = None
            y_length = None
            if self.task.in_out_same_size or self.task.all_out_same_size:
                x_length = self.task.shapes[self.task.n_train + example_num][1][0]
                y_length = self.task.shapes[self.task.n_train + example_num][1][1]

            solution_slice = self.best_crop(colors[example_num],
                                           x_mask[example_num],
                                           x_length,
                                           y_mask[example_num],
                                           y_length)
            uncertainty_slice = self.best_crop(uncertainties[example_num],
                                              x_mask[example_num],
                                              x_length,
                                              y_mask[example_num],
                                              y_length)

            solution_slices.append(solution_slice.cpu().numpy().tolist())
            uncertainty_values.append(float(np.mean(uncertainty_slice.cpu().numpy())))

        # Map colors
        for example in solution_slices:
            for row in example:
                for i, val in enumerate(row):
                    row[i] = self.task.colors[val]

        solution_slices = tuple(tuple(tuple(row) for row in example) for example in solution_slices)
        return solution_slices, np.mean(uncertainty_values)

    def _nucleus_sample(self, logits, p=0.9):
        """
        Nucleus (top-p) sampling for diverse solutions.

        Args:
            logits: Prediction logits (example, color, x, y)
            p: Cumulative probability threshold

        Returns:
            Sampled colors (example, x, y)
        """
        probs = torch.softmax(logits, dim=1)
        colors = torch.zeros(logits.shape[0], logits.shape[2], logits.shape[3], dtype=torch.long)

        for ex in range(logits.shape[0]):
            for x in range(logits.shape[2]):
                for y in range(logits.shape[3]):
                    pixel_probs = probs[ex, :, x, y]

                    # Sort by probability
                    sorted_probs, sorted_indices = torch.sort(pixel_probs, descending=True)

                    # Find cumulative threshold
                    cumsum = torch.cumsum(sorted_probs, dim=0)
                    cutoff_idx = torch.where(cumsum >= p)[0][0].item() + 1 if torch.any(cumsum >= p) else len(sorted_probs)

                    # Sample from top-p
                    top_probs = sorted_probs[:cutoff_idx]
                    top_indices = sorted_indices[:cutoff_idx]

                    # Renormalize and sample
                    top_probs = top_probs / top_probs.sum()
                    sampled_idx = np.random.choice(len(top_probs), p=top_probs.cpu().numpy())
                    colors[ex, x, y] = top_indices[sampled_idx]

        return colors


# Keep compatibility with existing code
def save_predictions(loggers, fname='predictions.npz'):
    """Saves solution score contributions and history of chosen solutions."""
    np.savez(fname,
             solution_contribution_logs=[logger.solution_contributions_log for logger in loggers],
             solution_picks_histories=[logger.solution_picks_history for logger in loggers])


def plot_accuracy(true_solution_hashes, fname='predictions.npz'):
    """Plots accuracy curve over training iterations."""
    stored_data = np.load(fname, allow_pickle=True)
    solution_picks_histories = stored_data['solution_picks_histories']

    n_tasks = len(solution_picks_histories)
    n_iterations = len(solution_picks_histories[0])

    correct = np.array([[
        int(any(hash_ == true_solution_hashes[task_num] for hash_ in solution_pair))
        for solution_pair in task_history
    ] for task_num, task_history in enumerate(solution_picks_histories)])

    accuracy_curve = correct.mean(axis=0)

    plt.figure()
    plt.plot(np.arange(n_iterations), accuracy_curve, 'k-')
    plt.savefig('accuracy_curve.pdf', bbox_inches='tight')
    plt.close()
