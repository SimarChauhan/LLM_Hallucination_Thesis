"""Dataset loading and preprocessing for TriviaQA."""

import logging
from typing import Iterator, Optional
from datasets import load_dataset

from .schemas import Question

logger = logging.getLogger(__name__)


def load_trivia_qa(
    subset: str = "rc",
    split: str = "validation",
    max_questions: Optional[int] = None
) -> Iterator[Question]:
    """
    Load TriviaQA dataset and yield Question objects.
    
    Args:
        subset: TriviaQA subset ("rc" has cleaner answers, "unfiltered" is larger)
        split: Dataset split ("train", "validation", "test")
        max_questions: Maximum number of questions to yield (None for all)
    
    Yields:
        Question objects with id, text, ground_truths, and category
    """
    logger.info(f"Loading TriviaQA dataset: subset={subset}, split={split}")
    
    # Load the dataset
    # TriviaQA structure: question, answer (dict with value, aliases, normalized_value, normalized_aliases)
    dataset = load_dataset("trivia_qa", subset, split=split, trust_remote_code=True)
    
    count = 0
    for idx, item in enumerate(dataset):
        if max_questions is not None and count >= max_questions:
            break
        
        # Extract question text
        question_text = item["question"]
        
        # Extract all acceptable answers
        # answer structure: {"value": str, "aliases": List[str], "normalized_value": str, "normalized_aliases": List[str]}
        answer_data = item["answer"]
        
        # Collect all answer variants
        ground_truths = []
        
        # Primary answer
        if answer_data.get("value"):
            ground_truths.append(answer_data["value"])
        
        # Aliases
        if answer_data.get("aliases"):
            ground_truths.extend(answer_data["aliases"])
        
        # Normalized versions (useful for matching)
        if answer_data.get("normalized_value"):
            ground_truths.append(answer_data["normalized_value"])
        
        if answer_data.get("normalized_aliases"):
            ground_truths.extend(answer_data["normalized_aliases"])
        
        # Deduplicate while preserving order
        seen = set()
        unique_truths = []
        for ans in ground_truths:
            ans_lower = ans.lower().strip()
            if ans_lower and ans_lower not in seen:
                seen.add(ans_lower)
                unique_truths.append(ans)
        
        if not unique_truths:
            logger.warning(f"Skipping question {idx}: no valid answers found")
            continue
        
        # Create Question object
        question = Question(
            id=f"tqa_{subset}_{split}_{idx}",
            text=question_text,
            ground_truths=unique_truths,
            category=None  # TriviaQA doesn't have explicit categories in the rc subset
        )
        
        yield question
        count += 1
    
    logger.info(f"Loaded {count} questions from TriviaQA")


def get_dataset_stats(
    subset: str = "rc",
    split: str = "validation"
) -> dict:
    """
    Get basic statistics about the dataset.
    
    Args:
        subset: TriviaQA subset
        split: Dataset split
    
    Returns:
        Dictionary with dataset statistics
    """
    dataset = load_dataset("trivia_qa", subset, split=split, trust_remote_code=True)
    
    total_questions = len(dataset)
    
    # Sample some answers to get stats
    sample_size = min(100, total_questions)
    avg_aliases = 0
    
    for i in range(sample_size):
        item = dataset[i]
        answer_data = item["answer"]
        num_aliases = len(answer_data.get("aliases", []))
        avg_aliases += num_aliases
    
    avg_aliases /= sample_size
    
    return {
        "total_questions": total_questions,
        "subset": subset,
        "split": split,
        "avg_aliases_per_question": round(avg_aliases, 2)
    }


def load_boolq(
    split: str = "validation",
    max_questions: Optional[int] = None,
) -> Iterator[Question]:
    """Load BoolQ as a complementary factual yes/no benchmark."""
    logger.info(f"Loading BoolQ dataset: split={split}")
    dataset = load_dataset("google/boolq", split=split, trust_remote_code=True)

    count = 0
    for idx, item in enumerate(dataset):
        if max_questions is not None and count >= max_questions:
            break
        question_text = (item.get("question") or "").strip()
        if not question_text:
            continue
        # Normalize into a QA-style prompt and simple gold aliases.
        answer_bool = bool(item.get("answer"))
        truth = "yes" if answer_bool else "no"
        ground_truths = [truth, truth.capitalize(), "true" if answer_bool else "false"]

        q = Question(
            id=f"boolq_{split}_{idx}",
            text=question_text,
            ground_truths=ground_truths,
            category="BoolQ",
        )
        yield q
        count += 1

    logger.info(f"Loaded {count} questions from BoolQ")


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    print("Testing TriviaQA loader...")
    stats = get_dataset_stats()
    print(f"Dataset stats: {stats}")
    
    print("\nFirst 3 questions:")
    for i, q in enumerate(load_trivia_qa(max_questions=3)):
        print(f"\n{i+1}. ID: {q.id}")
        print(f"   Question: {q.text[:100]}...")
        print(f"   Answers: {q.ground_truths[:5]}...")
