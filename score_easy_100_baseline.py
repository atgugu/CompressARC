"""
Script to score the baseline results on 100 easiest ARC tasks.
"""

import json
from typing import Tuple
import argparse
import sys

def score_submission(submission_file_name, solutions_file_name, include_task_scores=False) -> dict:
    """
    Score a submission against ground truth solutions.

    Args:
        submission_file_name (str): The file name of the submission file.
        solutions_file_name (str): The file name of the ground truth solutions.
        include_task_scores (bool, optional): Whether to include individual task scores. Defaults to False.

    Returns:
        dict: A dictionary containing the total score, total tasks scored, and optionally individual task scores.

    Reads a submission from file, scores it against the solutions, and returns the score.
    """
    # Open your submission & solutions file
    with open(submission_file_name, "r") as file:
        submission = json.load(file)

    with open(solutions_file_name, "r") as file:
        solutions = json.load(file)

    total_score = 0
    total_tasks = 0
    task_scores = {}
    tasks_correct = 0

    # Loop through each task in your submission to grade it
    for task_id, task_submission in submission.items():
        total_tasks += 1
        task_score = 0
        num_pairs = len(task_submission)

        # Go through each task pair. Most will only have 1
        for pair_index, pair_attempts in enumerate(task_submission):
            pair_correct = False

            # Look at both of your attempts
            for attempt_key, attempt in pair_attempts.items():

                # Check to see if one is correct
                if attempt == solutions[task_id][pair_index]:
                    pair_correct = True
                    break # If it is correct, log it and break the loop

            if pair_correct:
                task_score += 1

        # Get the average score across the sub-tasks/pairs
        task_score /= num_pairs

        # Add it to your total score
        total_score += task_score

        # Log it for that task
        task_scores[task_id] = task_score

        if task_score == 1.0:
            tasks_correct += 1

    result = {
        'total_score': total_score,
        'total_tasks_scored': total_tasks,
        'average_score': total_score / total_tasks if total_tasks > 0 else 0,
        'tasks_fully_correct': tasks_correct,
        'accuracy': tasks_correct / total_tasks if total_tasks > 0 else 0
    }

    if include_task_scores:
        result['task_scores'] = task_scores

    return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Score ARC task submissions')
    parser.add_argument('--submission', type=str, default='./submission_easy_100.json',
                       help='Submission file to score')
    parser.add_argument('--solutions', type=str, default='dataset/arc-agi_training_solutions.json',
                       help='Ground truth solutions file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for detailed results (optional)')

    args = parser.parse_args()

    submission_file_name = args.submission
    solutions_file_name = args.solutions

    print("Scoring ARC task results...")
    print(f"Submission file: {submission_file_name}")
    print(f"Solutions file: {solutions_file_name}")
    print()

    score = score_submission(submission_file_name, solutions_file_name, include_task_scores=True)

    print("=" * 60)
    print("ARC TASK EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Score: {score['total_score']:.2f}")
    print(f"Total Tasks: {score['total_tasks_scored']}")
    print(f"Average Score: {score['average_score']:.4f}")
    print(f"Tasks Fully Correct: {score['tasks_fully_correct']}")
    print(f"Accuracy (% fully correct): {score['accuracy']*100:.2f}%")
    print("=" * 60)

    # Show some individual task scores
    print("\nSample task scores (first 10):")
    for i, (task_id, task_score) in enumerate(list(score['task_scores'].items())[:10]):
        print(f"  {task_id}: {task_score:.2f}")

    # Save detailed results
    if args.output:
        output_file = args.output
    else:
        # Auto-generate output filename based on submission
        base_name = submission_file_name.replace('.json', '_results.json')
        output_file = base_name

    with open(output_file, 'w') as f:
        json.dump(score, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")
