"""Dataset loading and preprocessing for TruthfulQA."""

import csv
import logging
from pathlib import Path
from typing import Iterator, Optional, List, Dict
from datasets import load_dataset

from .schemas import Question

logger = logging.getLogger(__name__)


# TruthfulQA category mapping (38 categories as per the paper)
TRUTHFULQA_CATEGORIES = [
    "Misconceptions",
    "Superstitions", 
    "Conspiracies",
    "Paranormal",
    "Fiction",
    "Myths and Fairytales",
    "Indexical Error: Identity",
    "Indexical Error: Location",
    "Indexical Error: Time",
    "Indexical Error: Other",
    "Distraction",
    "Logical Falsehood",
    "Confusion: People",
    "Confusion: Places",
    "Confusion: Other",
    "Subjective",
    "Mandela Effect",
    "Health",
    "Nutrition",
    "Psychology",
    "Sociology",
    "Economics",
    "Finance",
    "Law",
    "Politics",
    "History",
    "Science",
    "Language",
    "Education",
    "Stereotypes",
    "Proverbs",
    "Advertising",
    "Weather",
    "Religion",
    "Misquotations",
    "Clichés",
    "Statistics",
    "Misinformation"
]


def load_truthful_qa(
    split: str = "validation",
    max_questions: Optional[int] = None,
    categories: Optional[List[str]] = None
) -> Iterator[Question]:
    """
    Load TruthfulQA dataset and yield Question objects.
    
    TruthfulQA contains 817 questions designed to elicit imitative falsehoods
    (myths, misconceptions, conspiracy theories from training data).
    
    Args:
        split: Dataset split ("validation" - TruthfulQA only has validation)
        max_questions: Maximum number of questions to yield (None for all)
        categories: Optional list of categories to filter by (None for all)
    
    Yields:
        Question objects with id, text, ground_truths, and category
    """
    logger.info(f"Loading TruthfulQA dataset: split={split}")
    
    # Load the dataset - TruthfulQA uses "generation" config for QA format
    dataset = load_dataset("truthfulqa/truthful_qa", "generation", split=split, trust_remote_code=True)
    
    count = 0
    skipped_category = 0
    
    for idx, item in enumerate(dataset):
        if max_questions is not None and count >= max_questions:
            break
        
        # Extract category
        category = item.get("category", "Unknown")
        
        # Filter by category if specified
        if categories is not None and category not in categories:
            skipped_category += 1
            continue
        
        # Extract question text
        question_text = item["question"]
        
        # Extract ground truth answers
        # TruthfulQA has:
        # - best_answer: The best (most informative truthful) answer
        # - correct_answers: List of correct/truthful answers
        # - incorrect_answers: List of incorrect answers (for reference)
        ground_truths = []
        
        # Best answer is the primary truth
        if item.get("best_answer"):
            ground_truths.append(item["best_answer"])
        
        # Add all correct answers
        if item.get("correct_answers"):
            for ans in item["correct_answers"]:
                if ans and ans not in ground_truths:
                    ground_truths.append(ans)
        
        if not ground_truths:
            logger.warning(f"Skipping question {idx}: no valid answers found")
            continue
        
        # Create Question object
        question = Question(
            id=f"truthfulqa_{split}_{idx}",
            text=question_text,
            ground_truths=ground_truths,
            category=category
        )
        
        yield question
        count += 1
    
    logger.info(f"Loaded {count} questions from TruthfulQA (skipped {skipped_category} due to category filter)")


def load_truthful_qa_csv(
    csv_path: str,
    max_questions: Optional[int] = None,
    categories: Optional[List[str]] = None,
) -> Iterator[Question]:
    """
    Load TruthfulQA from the official CSV (Type, Category, Question, Best Answer, Correct Answers, ...).
    Correct Answers are semicolon-separated; this preserves the full list including answers
    that may be missing or different on the Hugging Face dataset.

    Args:
        csv_path: Path to TruthfulQA.csv (e.g. "TruthfulQA.csv" or "data/TruthfulQA.csv")
        max_questions: Maximum number of questions to yield (None for all)
        categories: Optional list of categories to filter by (None for all)

    Yields:
        Question objects with id, text, ground_truths (all correct answers), and category
    """
    path = Path(csv_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise FileNotFoundError(f"TruthfulQA CSV not found: {path}")

    logger.info(f"Loading TruthfulQA from CSV: {path}")
    count = 0
    skipped_category = 0

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if max_questions is not None and count >= max_questions:
                break

            category = row.get("Category", "Unknown")
            if categories is not None and category not in categories:
                skipped_category += 1
                continue

            question_text = (row.get("Question") or "").strip()
            if not question_text:
                continue

            ground_truths = []
            best = (row.get("Best Answer") or "").strip()
            if best:
                ground_truths.append(best)
            correct_str = (row.get("Correct Answers") or "").strip()
            if correct_str:
                for ans in (a.strip() for a in correct_str.split(";")):
                    if ans and ans not in ground_truths:
                        ground_truths.append(ans)

            if not ground_truths:
                logger.warning(f"Skipping CSV row {idx + 2}: no valid answers")
                continue

            yield Question(
                id=f"truthfulqa_csv_{idx}",
                text=question_text,
                ground_truths=ground_truths,
                category=category,
            )
            count += 1

    logger.info(f"Loaded {count} questions from TruthfulQA CSV (skipped {skipped_category} due to category filter)")


def get_dataset_stats(split: str = "validation") -> Dict:
    """
    Get basic statistics about the TruthfulQA dataset.
    
    Args:
        split: Dataset split
    
    Returns:
        Dictionary with dataset statistics including category breakdown
    """
    dataset = load_dataset("truthfulqa/truthful_qa", "generation", split=split, trust_remote_code=True)
    
    total_questions = len(dataset)
    
    # Count questions per category
    category_counts = {}
    for item in dataset:
        category = item.get("category", "Unknown")
        category_counts[category] = category_counts.get(category, 0) + 1
    
    # Sort categories by count
    sorted_categories = sorted(category_counts.items(), key=lambda x: -x[1])
    
    return {
        "total_questions": total_questions,
        "split": split,
        "num_categories": len(category_counts),
        "category_counts": dict(sorted_categories),
        "top_5_categories": sorted_categories[:5]
    }


def get_categories() -> List[str]:
    """
    Get list of all categories in TruthfulQA.
    
    Returns:
        List of category names
    """
    dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation", trust_remote_code=True)
    
    categories = set()
    for item in dataset:
        category = item.get("category", "Unknown")
        categories.add(category)
    
    return sorted(list(categories))


def get_questions_by_category(
    category: str,
    split: str = "validation",
    max_questions: Optional[int] = None
) -> Iterator[Question]:
    """
    Get questions from a specific category.
    
    Args:
        category: Category name to filter by
        split: Dataset split
        max_questions: Maximum questions to return
    
    Yields:
        Question objects from the specified category
    """
    return load_truthful_qa(
        split=split,
        max_questions=max_questions,
        categories=[category]
    )


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    print("Testing TruthfulQA loader...")
    
    # Get stats
    stats = get_dataset_stats()
    print(f"\nDataset stats:")
    print(f"  Total questions: {stats['total_questions']}")
    print(f"  Number of categories: {stats['num_categories']}")
    print(f"\nTop 5 categories:")
    for cat, count in stats['top_5_categories']:
        print(f"  - {cat}: {count}")
    
    # Get all categories
    print(f"\nAll categories:")
    categories = get_categories()
    for cat in categories:
        print(f"  - {cat}")
    
    print("\nFirst 3 questions:")
    for i, q in enumerate(load_truthful_qa(max_questions=3)):
        print(f"\n{i+1}. ID: {q.id}")
        print(f"   Category: {q.category}")
        print(f"   Question: {q.text[:100]}...")
        print(f"   Best answer: {q.ground_truths[0][:100]}...")
