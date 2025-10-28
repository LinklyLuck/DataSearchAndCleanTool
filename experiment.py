"""
For performance testin,R^2 collection and batch testing only

Enhancements over original expierment.py:
1. Added error recovery
2. Integration with validation
3. Better logging
4. Configurable via CLI args
"""

import subprocess
import csv
import json
import time
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Runs batch experiments and collects metrics."""

    def __init__(
            self,
            queries_csv: Path,
            pipeline_script: Path,
            results_dir: Path,
            validator_script: Path = None,
            continue_on_error: bool = True
    ):
        self.queries_csv = queries_csv
        self.pipeline_script = pipeline_script
        self.results_dir = results_dir
        self.validator_script = validator_script
        self.continue_on_error = continue_on_error

        self.results_dir.mkdir(exist_ok=True, parents=True)

    def load_queries(self) -> List[str]:
        """Load queries from CSV file."""
        if not self.queries_csv.exists():
            raise FileNotFoundError(f"Queries CSV not found: {self.queries_csv}")

        queries = []
        with self.queries_csv.open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "NL Questions" in row and row["NL Questions"].strip():
                    queries.append(row["NL Questions"].strip())

        if not queries:
            raise ValueError("No queries found in CSV (need 'NL Questions' column)")

        logger.info(f"Loaded {len(queries)} queries from {self.queries_csv}")
        return queries

    def make_slug(self, idx: int, question: str, max_len: int = 40) -> str:
        """Create safe filename slug."""
        base = ''.join(ch if ch.isalnum() else '_' for ch in question.lower())
        base = base[:max_len] or 'query'
        return f"q{idx:03d}_{base}"

    def run_pipeline(self, idx: int, question: str) -> Dict[str, Any]:
        """Run pipeline for a single query."""
        slug = self.make_slug(idx, question)
        logger.info(f"[{idx}] Processing: {question[:80]}...")

        # Prepare config
        cfg_path = self.results_dir / f"{slug}_config.json"
        out_csv = self.results_dir / f"{slug}_merged.csv"
        graph_dot = self.results_dir / f"{slug}_graph.dot"

        config = {
            "question": question,
            "datasets": [
                "datas/**/*.{csv,json,ndjson,jsonl,xlsx,xls}",
                "datalake/**/*.{csv,json,ndjson,jsonl,xlsx,xls}"
            ],
            "out": str(out_csv),
            "graph_out": str(graph_dot),
        }

        cfg_path.write_text(json.dumps(config, indent=2, ensure_ascii=False))

        # Run pipeline
        t0 = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, str(self.pipeline_script), str(cfg_path)],
                capture_output=True,
                text=True,
                timeout=300  # 5 min timeout
            )
            elapsed = time.time() - t0

            # Parse R² scores from stdout
            r2_scores = {}
            merged_tables = []
            for line in proc.stdout.splitlines():
                if line.startswith("MERGED::"):
                    merged_tables = [t.strip() for t in line.split("::", 1)[1].split(",")]
                if line.strip().startswith("R2::"):
                    parts = line.split()
                    if len(parts) >= 3:
                        r2_scores[parts[1]] = float(parts[2])

            result = {
                "query": question,
                "slug": slug,
                "status": "success" if proc.returncode == 0 else "failed",
                "returncode": proc.returncode,
                "elapsed_s": round(elapsed, 2),
                "r2_scores": r2_scores,
                "merged_tables": merged_tables,
                "out_csv": str(out_csv) if out_csv.exists() else None,
                "graph_dot": str(graph_dot) if graph_dot.exists() else None,
                "stdout_tail": proc.stdout[-500:],
                "stderr_tail": proc.stderr[-500:] if proc.stderr else None,
            }

            # Run validation if available
            if self.validator_script and out_csv.exists():
                result["validation"] = self.run_validation(slug, out_csv)

            logger.info(f"[{idx}] ✓ Completed in {elapsed:.1f}s")
            return result

        except subprocess.TimeoutExpired:
            logger.error(f"[{idx}] ✗ Timeout (>5min)")
            return {
                "query": question,
                "slug": slug,
                "status": "timeout",
                "elapsed_s": time.time() - t0,
            }
        except Exception as e:
            logger.error(f"[{idx}] ✗ Error: {e}")
            return {
                "query": question,
                "slug": slug,
                "status": "error",
                "error": str(e),
            }

    def run_validation(self, slug: str, merged_csv: Path) -> Dict[str, Any]:
        """Run validation script."""
        try:
            report_path = self.results_dir / f"{slug}_validation.csv"
            proc = subprocess.run(
                [
                    sys.executable, str(self.validator_script),
                    "--csv", str(merged_csv),
                    "--out", str(report_path)
                ],
                capture_output=True,
                text=True,
                timeout=60
            )

            return {
                "report_csv": str(report_path) if report_path.exists() else None,
                "returncode": proc.returncode,
                "stdout_tail": proc.stdout[-300:],
            }
        except Exception as e:
            return {"error": str(e)}

    def run_all(self) -> List[Dict[str, Any]]:
        """Run experiments for all queries."""
        queries = self.load_queries()
        results = []

        for idx, query in enumerate(queries, 1):
            result = self.run_pipeline(idx, query)
            results.append(result)

            if not self.continue_on_error and result.get("status") != "success":
                logger.warning("Stopping due to error (continue_on_error=False)")
                break

        return results

    def save_summary(self, results: List[Dict[str, Any]]):
        """Save summary report."""
        summary_path = self.results_dir / "summary.json"
        summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        logger.info(f"Summary saved to {summary_path}")

        # Print statistics
        total = len(results)
        success = sum(1 for r in results if r.get("status") == "success")
        failed = sum(1 for r in results if r.get("status") == "failed")
        avg_time = sum(r.get("elapsed_s", 0) for r in results) / max(1, total)

        logger.info("=" * 60)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total queries:  {total}")
        logger.info(f"Success:        {success} ({success / total * 100:.1f}%)")
        logger.info(f"Failed:         {failed} ({failed / total * 100:.1f}%)")
        logger.info(f"Avg time:       {avg_time:.1f}s")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run batch experiments for data pipeline"
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("queries_sig.csv"),
        help="Path to queries CSV file"
    )
    parser.add_argument(
        "--pipeline",
        type=Path,
        default=Path("the_pipeline_v2.py"),
        help="Path to pipeline script"
    )
    parser.add_argument(
        "--validator",
        type=Path,
        default=Path("validate_output_dataset.py"),
        help="Path to validation script"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results_experiment"),
        help="Results directory"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Continue even if a query fails"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.queries.exists():
        logger.error(f"Queries file not found: {args.queries}")
        sys.exit(1)
    if not args.pipeline.exists():
        logger.error(f"Pipeline script not found: {args.pipeline}")
        sys.exit(1)

    # Run experiments
    runner = ExperimentRunner(
        queries_csv=args.queries,
        pipeline_script=args.pipeline,
        results_dir=args.results,
        validator_script=args.validator if args.validator.exists() else None,
        continue_on_error=args.continue_on_error
    )

    results = runner.run_all()
    runner.save_summary(results)


if __name__ == "__main__":
    main()