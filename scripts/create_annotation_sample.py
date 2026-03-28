#!/usr/bin/env python3
"""
Create a stratified annotation sample from evaluated results for human review.

Usage:
    python scripts/create_annotation_sample.py
    python scripts/create_annotation_sample.py --n 100 --format csv
    python scripts/create_annotation_sample.py --input data/results/evaluated/results_v2_eval.jsonl
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.annotation import (
    select_annotation_sample,
    export_annotation_sheet,
    load_human_annotations,
    compute_calibration_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Create stratified annotation sample for human review"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/results/evaluated/results_v2_eval.jsonl",
        help="Input evaluated JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/annotations",
        help="Output directory for annotation files",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=50,
        help="Number of records to sample (default: 50)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "jsonl"],
        default="csv",
        help="Output format (default: csv)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--stratify-by",
        type=str,
        default="correctness_match_type",
        help="Field to stratify sample on",
    )
    # Calibration mode: compare system vs human annotations
    parser.add_argument(
        "--calibrate",
        type=str,
        default=None,
        help="Path to completed annotation file for calibration (CSV or JSONL)",
    )

    args = parser.parse_args()

    if args.calibrate:
        # Calibration mode
        logger.info(f"Loading human annotations from {args.calibrate}")
        annotations = load_human_annotations(args.calibrate)
        metrics = compute_calibration_metrics(annotations)

        print("\n" + "=" * 60)
        print("CALIBRATION RESULTS: System vs Human Agreement")
        print("=" * 60)
        print(f"Total annotations:  {metrics['n_total']}")
        print(f"Valid (both have judgments): {metrics['n_valid']}")
        print(f"Accuracy:           {metrics['accuracy']:.1%}")
        print(f"Cohen's Kappa:      {metrics['cohens_kappa']:.4f}")
        cm = metrics["confusion_matrix"]
        print(f"\nConfusion Matrix:")
        print(f"  True Positives:   {cm['TP']}")
        print(f"  False Positives:  {cm['FP']}")
        print(f"  True Negatives:   {cm['TN']}")
        print(f"  False Negatives:  {cm['FN']}")

        if metrics["per_match_type"]:
            print(f"\nPer Match Type:")
            for mt, info in metrics["per_match_type"].items():
                print(f"  {mt:30s}  accuracy={info['accuracy']:.1%}  (n={info['n']})")

        # Save to JSON
        output_json = Path(args.output_dir) / "calibration_results.json"
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved calibration results to {output_json}")
        return

    # Sample mode
    input_path = Path(args.input)
    if not input_path.exists():
        # Try relative to project root
        alt = Path(__file__).parent.parent / args.input
        if alt.exists():
            input_path = alt
        else:
            logger.error(f"Input file not found: {args.input}")
            sys.exit(1)

    # Load records
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} records from {input_path}")

    # Sample
    sample = select_annotation_sample(
        records,
        n=args.n,
        stratify_by=args.stratify_by,
        seed=args.seed,
    )

    # Export
    timestamp = datetime.now().strftime("%Y%m%d")
    ext = "csv" if args.format == "csv" else "jsonl"
    output_path = str(Path(args.output_dir) / f"annotation_sample_{timestamp}.{ext}")

    export_annotation_sheet(sample, output_path, fmt=args.format)

    print(f"\n{'=' * 60}")
    print(f"ANNOTATION SAMPLE CREATED")
    print(f"{'=' * 60}")
    print(f"Records sampled: {len(sample)}")
    print(f"Output file:     {output_path}")
    print(f"Format:          {args.format}")
    print(f"\nInstructions for annotator:")
    print(f"  1. Open {output_path}")
    print(f"  2. For each record, review the question + ground truth + greedy answer")
    print(f"  3. Fill in 'human_correct' column: TRUE / FALSE / UNCLEAR")
    print(f"  4. Optionally add notes in 'human_notes'")
    print(f"  5. Save and run: python scripts/create_annotation_sample.py --calibrate {output_path}")


if __name__ == "__main__":
    main()
