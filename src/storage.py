"""Storage utilities for saving and loading results."""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Iterator, Optional, Set, Tuple, Dict, Any

import pandas as pd

from .schemas import ResultRecord

logger = logging.getLogger(__name__)


class ResultStorage:
    """Handles saving and loading of pipeline results."""
    
    FAILED_PAIRS_FILE = "failed_pairs.jsonl"
    RUN_METADATA_FILE = "run_metadata.json"
    RETRY_QUEUE_FILE = "retry_queue.jsonl"
    
    def __init__(self, results_dir: str, results_file: str = "results.jsonl"):
        """
        Initialize the storage handler.
        
        Args:
            results_dir: Directory to store results
            results_file: Name of the JSON lines file
        """
        self.results_dir = Path(results_dir)
        self.results_file = self.results_dir / results_file
        self.failed_pairs_file = self.results_dir / self.FAILED_PAIRS_FILE
        self.run_metadata_file = self.results_dir / self.RUN_METADATA_FILE
        self.retry_queue_file = self.results_dir / self.RETRY_QUEUE_FILE
        
        # Ensure directory exists
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def save_record(self, record: ResultRecord) -> None:
        """
        Append a single result record to the JSON lines file.
        
        Args:
            record: ResultRecord to save
        """
        with open(self.results_file, "a", encoding="utf-8") as f:
            json_str = json.dumps(record.to_dict(), ensure_ascii=False)
            f.write(json_str + "\n")
        
        logger.debug(f"Saved record for question {record.question_id}, model {record.model}")
    
    def load_records(self) -> Iterator[ResultRecord]:
        """
        Load all result records from the JSON lines file.
        
        Yields:
            ResultRecord objects
        """
        if not self.results_file.exists():
            return
        
        with open(self.results_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    yield ResultRecord.from_dict(data)
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing line {line_num}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error creating record from line {line_num}: {e}")
                    continue
    
    def log_failed_pair(self, question_id: str, model_name: str, error_message: str) -> None:
        """
        Log a failed (question_id, model) pair for later retry.
        
        Args:
            question_id: Question identifier
            model_name: Model name that failed
            error_message: Exception or error description
        """
        record = {
            "question_id": question_id,
            "model": model_name,
            "error": error_message,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        with open(self.failed_pairs_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.warning(f"Logged failed pair: {question_id}, {model_name}")
    
    def get_failed_pairs(self) -> List[Dict[str, Any]]:
        """
        Load all failed pairs from the log file.
        
        Returns:
            List of dicts with question_id, model, error, timestamp
        """
        if not self.failed_pairs_file.exists():
            return []
        pairs = []
        with open(self.failed_pairs_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        pairs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return pairs
    
    def write_run_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Write run metadata for reproducibility.
        
        Args:
            metadata: Dict with run_timestamp, config_path, models, dataset, judge, etc.
        """
        with open(self.run_metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Wrote run metadata to {self.run_metadata_file}")

    @staticmethod
    def write_jsonl_atomic(records: List[Dict[str, Any]], output_path: str) -> None:
        """Atomically write a JSONL file via temp-file + replace."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp_path, path)
    
    def get_completed_pairs(
        self,
        required_samples: Optional[int] = None,
        require_non_empty_greedy: bool = False,
        require_non_empty_stochastic: bool = False,
        reject_incomplete_flag: bool = False,
    ) -> Set[Tuple[str, str]]:
        """
        Get set of (question_id, model) pairs that qualify as completed.

        Optional validation flags make resume robust: malformed/partial rows
        are ignored so they can be recollected, while valid rows are never
        re-requested.

        Args:
            required_samples: If set, require at least this many stochastic samples.
            require_non_empty_greedy: Require non-empty greedy answer text.
            require_non_empty_stochastic: Require non-empty stochastic list and entries.
            reject_incomplete_flag: Reject rows explicitly marked as incomplete.

        Returns:
            Set of (question_id, model) tuples
        """
        completed: Set[Tuple[str, str]] = set()
        invalid_rows = 0

        for record in self.load_records():
            question_id = str(getattr(record, "question_id", "") or "").strip()
            model = str(getattr(record, "model", "") or "").strip()
            if not question_id or not model:
                invalid_rows += 1
                continue

            if require_non_empty_greedy:
                greedy_answer = str(getattr(record, "greedy_answer", "") or "").strip()
                if not greedy_answer:
                    invalid_rows += 1
                    continue

            stochastic_answers = getattr(record, "stochastic_answers", None)
            stochastic_list: List[Any] = (
                stochastic_answers if isinstance(stochastic_answers, list) else []
            )

            if require_non_empty_stochastic:
                if not stochastic_list:
                    invalid_rows += 1
                    continue
                if any(not str(sample or "").strip() for sample in stochastic_list):
                    invalid_rows += 1
                    continue

            if required_samples is not None and len(stochastic_list) < int(required_samples):
                invalid_rows += 1
                continue

            if reject_incomplete_flag and bool(getattr(record, "is_incomplete", False)):
                invalid_rows += 1
                continue

            completed.add((question_id, model))

        if invalid_rows:
            logger.info(
                "Resume scan ignored %d malformed/incomplete rows; those pairs will be recollected.",
                invalid_rows,
            )
        return completed

    def enqueue_retry(self, payload: Dict[str, Any]) -> None:
        """Append one retry payload to the durable retry queue."""
        with open(self.retry_queue_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def load_retry_queue(self) -> List[Dict[str, Any]]:
        """Load queued retry payloads from disk."""
        if not self.retry_queue_file.exists():
            return []
        items: List[Dict[str, Any]] = []
        with open(self.retry_queue_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return items

    def clear_retry_queue(self) -> None:
        """Clear the retry queue file if it exists."""
        if self.retry_queue_file.exists():
            self.retry_queue_file.unlink()
    
    def count_records(self) -> int:
        """
        Count total number of records.
        
        Returns:
            Number of records in the file
        """
        count = 0
        for _ in self.load_records():
            count += 1
        return count
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Load all records into a pandas DataFrame.
        
        Returns:
            DataFrame with one row per record
        """
        records = []
        
        for record in self.load_records():
            record_dict = record.to_dict()
            
            # Flatten equivalence_stats
            if record_dict.get("equivalence_stats"):
                stats = record_dict.pop("equivalence_stats")
                record_dict["equiv_num_same"] = stats["num_same"]
                record_dict["equiv_num_different"] = stats["num_different"]
                record_dict["equiv_num_unclear"] = stats["num_unclear"]
                record_dict["equiv_total"] = stats["total"]
            
            records.append(record_dict)
        
        if not records:
            return pd.DataFrame()
        
        return pd.DataFrame(records)
    
    def export_to_parquet(self, parquet_file: Optional[str] = None) -> str:
        """
        Export results to Parquet format for efficient analysis.
        
        Args:
            parquet_file: Output file name (default: results.parquet)
        
        Returns:
            Path to the created Parquet file
        """
        if parquet_file is None:
            parquet_file = "results.parquet"
        
        output_path = self.results_dir / parquet_file
        
        df = self.to_dataframe()
        
        if df.empty:
            logger.warning("No records to export")
            return str(output_path)
        
        df.to_parquet(output_path, index=False)
        logger.info(f"Exported {len(df)} records to {output_path}")
        
        return str(output_path)
    
    def get_summary_stats(self) -> dict:
        """
        Get summary statistics about stored results.
        
        Returns:
            Dictionary with summary statistics
        """
        df = self.to_dataframe()
        
        if df.empty:
            return {"total_records": 0}
        
        stats = {
            "total_records": len(df),
            "unique_questions": df["question_id"].nunique(),
            "unique_models": df["model"].nunique(),
            "models": df["model"].unique().tolist(),
        }
        
        # Correctness stats
        stats["correct_count"] = df["greedy_correct"].sum()
        stats["incorrect_count"] = (~df["greedy_correct"]).sum()
        stats["accuracy"] = stats["correct_count"] / len(df) if len(df) > 0 else 0
        
        # Error type stats (for incorrect answers)
        incorrect_df = df[~df["greedy_correct"]]
        if not incorrect_df.empty and "error_label_0.9" in incorrect_df.columns:
            label_counts = incorrect_df["error_label_0.9"].value_counts().to_dict()
            stats["error_labels_0.9"] = label_counts
        
        return stats


def load_results_from_file(filepath: str) -> List[ResultRecord]:
    """
    Load results from a JSON lines file.
    
    Args:
        filepath: Path to the JSON lines file
    
    Returns:
        List of ResultRecord objects
    """
    records = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                records.append(ResultRecord.from_dict(data))
    
    return records


if __name__ == "__main__":
    import tempfile
    from datetime import datetime
    
    from .schemas import EquivalenceStats
    
    print("Testing storage module...\n")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = ResultStorage(tmpdir)
        
        # Create test records
        test_record = ResultRecord(
            question_id="test_001",
            question="What is the capital of France?",
            ground_truth=["Paris", "paris"],
            model="test-model",
            greedy_answer="Lyon",
            greedy_correct=False,
            correctness_match_type=None,
            stochastic_answers=["Lyon", "Lyon", "Paris"],
            equivalence_results=["same", "same", "different"],
            equivalence_stats=EquivalenceStats(
                num_same=2,
                num_different=1,
                num_unclear=0,
                total=3
            ),
            equivalence_ratio=0.67,
            error_label_1_0="inconsistent_error",
            error_label_0_9="inconsistent_error",
            error_label_0_8="inconsistent_error",
            error_label_0_7="inconsistent_error",
        )
        
        # Save record
        storage.save_record(test_record)
        print(f"Saved record to {storage.results_file}")
        
        # Load records
        loaded = list(storage.load_records())
        print(f"Loaded {len(loaded)} records")
        
        # Check completed pairs
        completed = storage.get_completed_pairs()
        print(f"Completed pairs: {completed}")
        
        # Get stats
        stats = storage.get_summary_stats()
        print(f"Summary stats: {stats}")
        
        # Test DataFrame export
        df = storage.to_dataframe()
        print(f"\nDataFrame shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
