from __future__ import annotations
import csv
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import polars as pl

# Visualization
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("[WARN] plotly not installed. HTML visualization will be limited.")


# ============================================================================
# Data Classes for Metrics
# ============================================================================

@dataclass
class DataQualityMetrics:
    """Data Quality Assessment"""
    completeness_score: float  # 0-1: % of non-null cells
    consistency_score: float  # 0-1: type consistency
    accuracy_score: float  # 0-1: estimated via heuristics

    total_cells: int
    non_null_cells: int
    null_cells: int
    null_rate: float

    numeric_columns: int
    text_columns: int
    mixed_columns: int

    duplicate_rows: int
    duplicate_rate: float

    column_quality: Dict[str, Dict[str, float]]  # per-column stats


@dataclass
class MergeQualityMetrics:
    """Merge/Join Quality Assessment"""
    join_accuracy_score: float  # 0-1: estimated quality
    data_retention_rate: float  # rows retained / total input rows

    total_input_rows: int
    total_output_rows: int
    rows_retained: int
    rows_lost: int

    join_keys_used: List[str]
    join_key_coverage: float  # % of files with join keys

    safe_mode_rows: int  # inner join result
    universal_mode_rows: int  # full outer join result
    mode_difference: int

    blowup_ratio: float  # output/input ratio
    is_blowup_safe: bool  # < 1.8x


@dataclass
class SchemaAlignmentMetrics:
    """Schema Mapping Quality"""
    mapping_accuracy_score: float  # 0-1: estimated

    total_source_columns: int
    total_target_features: int
    mapped_columns: int
    unmapped_columns: int

    mapping_coverage: float  # mapped / total_target

    per_file_mapping: List[Dict[str, Any]]  # detailed per-file stats

    alias_rule_hits: int  # how many times alias rules corrected mapping
    llm_mapping_attempts: int


@dataclass
class ERMetrics:
    """Entity Resolution Effectiveness"""
    enabled: bool

    deduplication_rate: float  # (input - output) / input

    input_rows: int
    output_rows: int
    rows_merged: int

    estimated_precision: float  # 0-1: heuristic
    estimated_recall: float  # 0-1: heuristic
    f1_score: float

    entity_clusters: int  # number of unique entities after ER
    avg_cluster_size: float

    provider_name: str
    sample_size: int


@dataclass
class MVIMetrics:
    """Missing Value Imputation Effectiveness"""
    enabled: bool

    fill_rate: float  # cells filled / total missing cells

    total_missing_before: int
    total_filled: int
    total_missing_after: int

    columns_updated: int
    rows_affected: int

    fill_rate_by_column: Dict[str, float]

    provider_name: str
    source_distribution: Dict[str, int]  # Wikipedia: 10, TMDb: 5, etc.

    sample_size: int


@dataclass
class PerformanceMetrics:
    """Performance Statistics"""
    total_time_seconds: float

    time_breakdown: Dict[str, float]  # scan, map, join, er, mvi, export

    peak_memory_mb: float
    avg_memory_mb: float

    files_processed: int
    avg_time_per_file: float

    llm_calls: int
    llm_time_seconds: float
    avg_llm_latency: float


@dataclass
class ValidationReport:
    """Complete Validation Report"""
    # Metadata
    report_id: str
    timestamp: str
    query: str
    output_file: str

    # Metrics
    data_quality: DataQualityMetrics
    merge_quality: MergeQualityMetrics
    schema_alignment: SchemaAlignmentMetrics
    er_metrics: ERMetrics
    mvi_metrics: MVIMetrics
    performance: PerformanceMetrics

    # Aggregate Score (0-100)
    overall_score: float

    # Diagnostics
    warnings: List[str]
    errors: List[str]


# ============================================================================
# Validation Engine
# ============================================================================

class ValidationEngine:
    """Generates comprehensive validation reports"""

    def __init__(self, merged_csv: Path, meta_json: Path):
        self.merged_csv = merged_csv
        self.meta_json = meta_json

        self.df: Optional[pl.DataFrame] = None
        self.meta: Dict[str, Any] = {}

        self.warnings: List[str] = []
        self.errors: List[str] = []

    def load_data(self):
        """Load merged CSV and metadata"""
        # Load CSV
        csv_read_attempts: List[Tuple[str, Dict[str, Any], str]] = [
            ("polars", {}, "default settings"),
            ("polars", {"infer_schema_length": 0}, "full-file schema inference"),
            (
                "polars",
                {"infer_schema_length": 0, "dtypes": pl.Utf8},
                "forcing all columns to strings",
            ),
            ("python", {}, "python csv fallback for nested data"),
        ]

        last_error: Optional[Exception] = None
        for engine, kwargs, description in csv_read_attempts:
            try:
                if engine == "polars":
                    self.df = pl.read_csv(self.merged_csv, **kwargs)
                else:
                    with self.merged_csv.open("r", encoding="utf-8", newline="") as fh:
                        reader = csv.DictReader(fh)
                        if reader.fieldnames is None:
                            raise ValueError("CSV file missing header row for python fallback")
                        rows = list(reader)
                    self.df = pl.from_dicts(rows)
                print(f"[ValidationEngine] Loaded CSV with {description}: {self.df.shape}")
                break
            except Exception as e:
                last_error = e
                print(
                    f"[ValidationEngine] CSV load failed using {description}: {e}. "
                    "Retrying with fallback strategy..."
                )
        else:
            error_message = f"Failed to load CSV after retries: {last_error}"
            self.errors.append(error_message)
            raise last_error

        # Load metadata
        try:
            self.meta = json.loads(self.meta_json.read_text(encoding='utf-8'))
            print(f"[ValidationEngine] Loaded metadata")
        except Exception as e:
            self.errors.append(f"Failed to load metadata: {e}")
            raise

    def compute_data_quality(self) -> DataQualityMetrics:
        """Compute data quality metrics"""
        if self.df is None:
            raise ValueError("Data not loaded")

        df = self.df
        n_rows, n_cols = df.shape
        total_cells = n_rows * n_cols

        # Null analysis
        null_counts = [df[col].null_count() for col in df.columns]
        total_null = sum(null_counts)
        non_null = total_cells - total_null
        null_rate = total_null / max(1, total_cells)

        completeness = 1.0 - null_rate

        # Type analysis
        numeric_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt.is_numeric()]
        text_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt == pl.Utf8]
        mixed_cols = len(df.columns) - len(numeric_cols) - len(text_cols)

        # Consistency: check if numeric columns have reasonable values
        consistency_scores = []
        for col in numeric_cols:
            try:
                s = df[col]
                if s.null_count() < len(s):
                    # Check for outliers (simple heuristic)
                    vals = s.drop_nulls()
                    if len(vals) > 0:
                        mean = vals.mean()
                        std = vals.std()
                        if std is not None and std > 0:
                            outliers = ((vals - mean).abs() > 3 * std).sum()
                            consistency_scores.append(1.0 - outliers / len(vals))
                        else:
                            consistency_scores.append(1.0)
            except:
                pass

        consistency = np.mean(consistency_scores) if consistency_scores else 0.5

        # Accuracy: heuristic based on completeness and consistency
        accuracy = (completeness + consistency) / 2.0

        # Duplicates
        try:
            dup_count = df.height - df.unique().height
            dup_rate = dup_count / max(1, df.height)
        except:
            dup_count = 0
            dup_rate = 0.0

        # Per-column quality
        column_quality = {}
        for col in df.columns:
            null_pct = df[col].null_count() / max(1, len(df))
            unique_pct = df[col].n_unique() / max(1, len(df))

            column_quality[col] = {
                "null_rate": round(null_pct, 4),
                "unique_rate": round(unique_pct, 4),
                "dtype": str(df[col].dtype)
            }

        return DataQualityMetrics(
            completeness_score=round(completeness, 4),
            consistency_score=round(consistency, 4),
            accuracy_score=round(accuracy, 4),
            total_cells=total_cells,
            non_null_cells=non_null,
            null_cells=total_null,
            null_rate=round(null_rate, 4),
            numeric_columns=len(numeric_cols),
            text_columns=len(text_cols),
            mixed_columns=mixed_cols,
            duplicate_rows=dup_count,
            duplicate_rate=round(dup_rate, 4),
            column_quality=column_quality
        )

    def compute_merge_quality(self) -> MergeQualityMetrics:
        """Compute merge/join quality metrics"""
        diagnostics = self.meta.get("diagnostics", [])

        # Total input rows
        total_input = sum(d.get("rows_raw", 0) for d in diagnostics)

        # Output rows
        output_rows = self.df.height if self.df is not None else 0

        rows_retained = output_rows
        rows_lost = max(0, total_input - output_rows)
        retention_rate = output_rows / max(1, total_input)

        # Join keys
        join_keys = self.meta.get("join_keys", [])
        files_with_keys = sum(1 for d in diagnostics if d.get("join_keys_used"))
        join_key_coverage = files_with_keys / max(1, len(diagnostics))

        # Safe vs Universal
        safe_snapshot = self.meta.get("safe_inner_snapshot", {})
        safe_rows = safe_snapshot.get("rows", 0)
        universal_rows = output_rows
        mode_diff = abs(universal_rows - safe_rows)

        # Blowup ratio
        blowup = output_rows / max(1, total_input)
        is_safe = blowup <= 1.8

        # Join accuracy score (heuristic)
        # Good if: retention high, blowup low, join keys present
        join_accuracy = min(1.0, (retention_rate + (1.0 - min(1.0, blowup - 1.0)) + join_key_coverage) / 3.0)

        return MergeQualityMetrics(
            join_accuracy_score=round(join_accuracy, 4),
            data_retention_rate=round(retention_rate, 4),
            total_input_rows=total_input,
            total_output_rows=output_rows,
            rows_retained=rows_retained,
            rows_lost=rows_lost,
            join_keys_used=join_keys,
            join_key_coverage=round(join_key_coverage, 4),
            safe_mode_rows=safe_rows,
            universal_mode_rows=universal_rows,
            mode_difference=mode_diff,
            blowup_ratio=round(blowup, 4),
            is_blowup_safe=is_safe
        )

    def compute_schema_alignment(self) -> SchemaAlignmentMetrics:
        """Compute schema mapping quality"""
        diagnostics = self.meta.get("diagnostics", [])

        total_source_cols = 0
        total_mapped = 0
        per_file = []

        for d in diagnostics:
            cols_after = d.get("columns_after_map", [])
            total_source_cols += len(cols_after)
            total_mapped += len(cols_after)

            per_file.append({
                "file": d.get("file", "unknown"),
                "columns_mapped": len(cols_after),
                "join_keys": d.get("join_keys_used", [])
            })

        target_features = len(self.df.columns) if self.df is not None else 0

        mapping_coverage = total_mapped / max(1, target_features * len(diagnostics))

        # Mapping accuracy: heuristic based on coverage
        mapping_accuracy = min(1.0, mapping_coverage)

        # Alias rule usage (if available in meta)
        alias_hits = self.meta.get("alias_rule_hits", 0)
        llm_attempts = len(diagnostics)

        return SchemaAlignmentMetrics(
            mapping_accuracy_score=round(mapping_accuracy, 4),
            total_source_columns=total_source_cols,
            total_target_features=target_features,
            mapped_columns=total_mapped,
            unmapped_columns=max(0, total_source_cols - total_mapped),
            mapping_coverage=round(mapping_coverage, 4),
            per_file_mapping=per_file,
            alias_rule_hits=alias_hits,
            llm_mapping_attempts=llm_attempts
        )

    def compute_er_metrics(self) -> ERMetrics:
        """Compute Entity Resolution metrics"""
        enrich = self.meta.get("enrich", {})

        # Check if ER was enabled
        enabled = enrich.get("provider") is not None or self.meta.get("entity_report") is not None

        if not enabled:
            return ERMetrics(
                enabled=False,
                deduplication_rate=0.0,
                input_rows=0,
                output_rows=0,
                rows_merged=0,
                estimated_precision=0.0,
                estimated_recall=0.0,
                f1_score=0.0,
                entity_clusters=0,
                avg_cluster_size=0.0,
                provider_name="None",
                sample_size=0
            )

        sample_n = enrich.get("sample_n", 0)

        # Try to parse ER report if available
        er_report_path = self.meta.get("entity_report")
        rows_merged = 0
        entity_clusters = 0

        if er_report_path and Path(er_report_path).exists():
            try:
                er_df = pl.read_csv(er_report_path)
                # Assuming report has "cluster_id", "records_merged"
                if "records_merged" in er_df.columns:
                    rows_merged = er_df["records_merged"].sum()
                entity_clusters = len(er_df)
            except:
                pass

        input_rows = sample_n
        output_rows = max(1, input_rows - rows_merged)
        dedup_rate = rows_merged / max(1, input_rows)

        # Precision/Recall: heuristic
        # High dedup rate suggests good recall
        # We assume precision is high (LLM-based ER is conservative)
        estimated_recall = min(1.0, dedup_rate * 2.0)  # heuristic
        estimated_precision = 0.9  # assume high precision for LLM-based ER

        f1 = 2 * estimated_precision * estimated_recall / max(0.01, estimated_precision + estimated_recall)

        avg_cluster = rows_merged / max(1, entity_clusters) if entity_clusters > 0 else 0

        return ERMetrics(
            enabled=True,
            deduplication_rate=round(dedup_rate, 4),
            input_rows=input_rows,
            output_rows=output_rows,
            rows_merged=rows_merged,
            estimated_precision=round(estimated_precision, 4),
            estimated_recall=round(estimated_recall, 4),
            f1_score=round(f1, 4),
            entity_clusters=entity_clusters,
            avg_cluster_size=round(avg_cluster, 2),
            provider_name=enrich.get("provider", "LLM-ER"),
            sample_size=sample_n
        )

    def compute_mvi_metrics(self) -> MVIMetrics:
        """Compute Missing Value Imputation metrics"""
        enrich = self.meta.get("enrich", {})

        # Check if MVI was enabled
        mvi_report_path = self.meta.get("entity_report")  # may contain MVI changes
        enrich_summary_path = self.meta.get("enrich_summary_csv")

        enabled = enrich_summary_path is not None or mvi_report_path is not None

        if not enabled:
            return MVIMetrics(
                enabled=False,
                fill_rate=0.0,
                total_missing_before=0,
                total_filled=0,
                total_missing_after=0,
                columns_updated=0,
                rows_affected=0,
                fill_rate_by_column={},
                provider_name="None",
                source_distribution={},
                sample_size=0
            )

        sample_n = enrich.get("sample_n", 0)

        # Parse enrichment summary
        fill_by_col = {}
        total_filled = 0
        cols_updated = 0

        if enrich_summary_path and Path(enrich_summary_path).exists():
            try:
                summary_df = pl.read_csv(enrich_summary_path)
                for row in summary_df.iter_rows(named=True):
                    col = row.get("column", "unknown")
                    filled = row.get("filled_cells", 0)
                    ratio = row.get("filled_ratio", 0.0)

                    fill_by_col[col] = ratio
                    total_filled += filled
                    cols_updated += 1
            except:
                pass

        # Parse MVI report for source distribution
        source_dist = {}
        rows_affected = 0

        if mvi_report_path and Path(mvi_report_path).exists():
            try:
                mvi_df = pl.read_csv(mvi_report_path)
                rows_affected = len(mvi_df["row_index"].unique()) if "row_index" in mvi_df.columns else 0

                # Heuristic: assume Wikipedia for most, TMDb/OMDb for movies
                # This would need actual tracking in the pipeline
                source_dist = {"Wikipedia": rows_affected}  # placeholder
            except:
                pass

        # Estimate missing before/after
        total_missing_before = sample_n * cols_updated  # rough estimate
        total_missing_after = max(0, total_missing_before - total_filled)
        fill_rate = total_filled / max(1, total_missing_before)

        return MVIMetrics(
            enabled=True,
            fill_rate=round(fill_rate, 4),
            total_missing_before=total_missing_before,
            total_filled=total_filled,
            total_missing_after=total_missing_after,
            columns_updated=cols_updated,
            rows_affected=rows_affected,
            fill_rate_by_column=fill_by_col,
            provider_name=enrich.get("provider", "Multi-API"),
            source_distribution=source_dist,
            sample_size=sample_n
        )

    def compute_performance(self) -> PerformanceMetrics:
        """Compute performance metrics"""
        # Note: This requires timing data to be logged in meta
        # For now, we'll use placeholder values

        diagnostics = self.meta.get("diagnostics", [])
        n_files = len(diagnostics)

        # Placeholder timing
        time_breakdown = {
            "scan": 0.0,
            "map": 0.0,
            "join": 0.0,
            "er": 0.0,
            "mvi": 0.0,
            "export": 0.0
        }

        total_time = sum(time_breakdown.values())
        avg_per_file = total_time / max(1, n_files)

        # Memory: placeholder
        peak_mem = 0.0
        avg_mem = 0.0

        # LLM calls: estimate from diagnostics
        llm_calls = n_files * 2  # rough estimate (map + join)
        llm_time = 0.0
        avg_latency = llm_time / max(1, llm_calls)

        return PerformanceMetrics(
            total_time_seconds=total_time,
            time_breakdown=time_breakdown,
            peak_memory_mb=peak_mem,
            avg_memory_mb=avg_mem,
            files_processed=n_files,
            avg_time_per_file=round(avg_per_file, 2),
            llm_calls=llm_calls,
            llm_time_seconds=llm_time,
            avg_llm_latency=round(avg_latency, 4)
        )

    def compute_overall_score(
            self,
            dq: DataQualityMetrics,
            mq: MergeQualityMetrics,
            sa: SchemaAlignmentMetrics,
            er: ERMetrics,
            mvi: MVIMetrics
    ) -> float:
        """Compute aggregate quality score (0-100)"""
        weights = {
            "data_quality": 0.3,
            "merge_quality": 0.3,
            "schema_alignment": 0.2,
            "er": 0.1,
            "mvi": 0.1
        }

        score = (
                        dq.accuracy_score * weights["data_quality"] +
                        mq.join_accuracy_score * weights["merge_quality"] +
                        sa.mapping_accuracy_score * weights["schema_alignment"] +
                        er.f1_score * weights["er"] +
                        mvi.fill_rate * weights["mvi"]
                ) * 100

        return round(score, 2)

    def generate_report(self) -> ValidationReport:
        """Generate complete validation report"""
        self.load_data()

        print("[ValidationEngine] Computing metrics...")

        dq = self.compute_data_quality()
        mq = self.compute_merge_quality()
        sa = self.compute_schema_alignment()
        er = self.compute_er_metrics()
        mvi = self.compute_mvi_metrics()
        perf = self.compute_performance()

        overall = self.compute_overall_score(dq, mq, sa, er, mvi)

        report = ValidationReport(
            report_id=f"vldb_{int(time.time())}",
            timestamp=datetime.now().isoformat(),
            query=self.meta.get("query", ""),
            output_file=str(self.merged_csv),
            data_quality=dq,
            merge_quality=mq,
            schema_alignment=sa,
            er_metrics=er,
            mvi_metrics=mvi,
            performance=perf,
            overall_score=overall,
            warnings=self.warnings,
            errors=self.errors
        )

        print(f"[ValidationEngine] Overall Score: {overall}/100")

        return report


# ============================================================================
# Report Exporters
# ============================================================================

class CSVExporter:
    """Export report to structured CSV"""

    @staticmethod
    def export(report: ValidationReport, output_path: Path):
        """Export to CSV with multiple sheets as separate CSVs"""
        base = output_path.parent / output_path.stem

        # Summary
        summary = {
            "report_id": report.report_id,
            "timestamp": report.timestamp,
            "query": report.query,
            "overall_score": report.overall_score,
            "data_quality_score": report.data_quality.accuracy_score,
            "merge_quality_score": report.merge_quality.join_accuracy_score,
            "schema_alignment_score": report.schema_alignment.mapping_accuracy_score,
            "er_enabled": report.er_metrics.enabled,
            "er_dedup_rate": report.er_metrics.deduplication_rate,
            "mvi_enabled": report.mvi_metrics.enabled,
            "mvi_fill_rate": report.mvi_metrics.fill_rate,
            "total_rows": report.data_quality.total_cells // max(1, len(report.data_quality.column_quality)),
            "total_columns": len(report.data_quality.column_quality),
            "warnings": len(report.warnings),
            "errors": len(report.errors)
        }

        pl.DataFrame([summary]).write_csv(f"{base}_summary.csv")

        # Data Quality Details
        dq_data = []
        for col, stats in report.data_quality.column_quality.items():
            dq_data.append({
                "column": col,
                "null_rate": stats["null_rate"],
                "unique_rate": stats["unique_rate"],
                "dtype": stats["dtype"]
            })

        if dq_data:
            pl.DataFrame(dq_data).write_csv(f"{base}_data_quality.csv")

        # Schema Alignment Details
        if report.schema_alignment.per_file_mapping:
            pl.DataFrame(report.schema_alignment.per_file_mapping).write_csv(f"{base}_schema_alignment.csv")

        print(f"[CSVExporter] Exported to {base}_*.csv")


class HTMLExporter:
    """Export report to interactive HTML with visualizations"""

    @staticmethod
    def export(report: ValidationReport, output_path: Path):
        """Generate HTML report with Plotly charts"""
        if not HAS_PLOTLY:
            print("[HTMLExporter] Plotly not available. Generating minimal HTML.")
            HTMLExporter._export_minimal(report, output_path)
            return

        HTMLExporter._export_full(report, output_path)

    @staticmethod
    def _export_minimal(report: ValidationReport, output_path: Path):
        """Minimal HTML without visualizations"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Validation Report - {report.report_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        .metric {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
        .score {{ font-size: 48px; font-weight: bold; color: #007bff; }}
    </style>
</head>
<body>
    <h1>VLDB Validation Report</h1>
    <div class="metric">
        <div class="score">{report.overall_score}/100</div>
        <p>Overall Quality Score</p>
    </div>
    <h2>Summary</h2>
    <ul>
        <li>Query: {report.query}</li>
        <li>Timestamp: {report.timestamp}</li>
        <li>Data Quality: {report.data_quality.accuracy_score:.2%}</li>
        <li>Merge Quality: {report.merge_quality.join_accuracy_score:.2%}</li>
        <li>Schema Alignment: {report.schema_alignment.mapping_accuracy_score:.2%}</li>
    </ul>
</body>
</html>
"""
        output_path.write_text(html, encoding='utf-8')
        print(f"[HTMLExporter] Minimal HTML -> {output_path}")

    @staticmethod
    def _export_full(report: ValidationReport, output_path: Path):
        """Full HTML with Plotly visualizations"""

        # Create visualizations
        figures = HTMLExporter._create_visualizations(report)

        # Combine into HTML
        html_parts = [HTMLExporter._html_header(report)]

        for fig in figures:
            html_parts.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))

        html_parts.append(HTMLExporter._html_footer())

        html = "\n".join(html_parts)
        output_path.write_text(html, encoding='utf-8')

        print(f"[HTMLExporter] Full HTML -> {output_path}")

    @staticmethod
    def _html_header(report: ValidationReport) -> str:
        """HTML header"""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>VLDB Validation Report - {report.report_id}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: white;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 36px;
        }}
        .score-card {{
            display: flex;
            justify-content: space-around;
            margin: 30px 0;
        }}
        .score-item {{
            text-align: center;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            min-width: 150px;
        }}
        .score-value {{
            font-size: 48px;
            font-weight: bold;
            color: #667eea;
        }}
        .score-label {{
            margin-top: 10px;
            color: #666;
            font-size: 14px;
        }}
        .section {{
            margin: 40px 0;
        }}
        .section h2 {{
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #667eea;
            color: white;
        }}
        .warning {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 10px 0;
        }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>🎓 VLDB-Grade Validation Report</h1>
        <p><strong>Report ID:</strong> {report.report_id}</p>
        <p><strong>Timestamp:</strong> {report.timestamp}</p>
        <p><strong>Query:</strong> {report.query}</p>
    </div>

    <div class="score-card">
        <div class="score-item">
            <div class="score-value">{report.overall_score}</div>
            <div class="score-label">Overall Score</div>
        </div>
        <div class="score-item">
            <div class="score-value">{report.data_quality.completeness_score:.2%}</div>
            <div class="score-label">Completeness</div>
        </div>
        <div class="score-item">
            <div class="score-value">{report.merge_quality.data_retention_rate:.2%}</div>
            <div class="score-label">Data Retention</div>
        </div>
        <div class="score-item">
            <div class="score-value">{report.merge_quality.join_accuracy_score:.2%}</div>
            <div class="score-label">Join Accuracy</div>
        </div>
    </div>
"""

    @staticmethod
    def _html_footer() -> str:
        """HTML footer"""
        return """
    <div class="section">
        <p style="text-align: center; color: #999; margin-top: 50px;">
            Generated by VLDB Validation Engine | Academic-Grade Data Integration Assessment
        </p>
    </div>
</div>
</body>
</html>
"""

    @staticmethod
    def _create_visualizations(report: ValidationReport) -> List[go.Figure]:
        """Create all visualizations"""
        figures = []

        # 1. Overall Metrics Radar Chart
        fig1 = go.Figure()

        categories = ['Data Quality', 'Merge Quality', 'Schema Alignment', 'ER (F1)', 'MVI (Fill)']
        values = [
            report.data_quality.accuracy_score * 100,
            report.merge_quality.join_accuracy_score * 100,
            report.schema_alignment.mapping_accuracy_score * 100,
            report.er_metrics.f1_score * 100 if report.er_metrics.enabled else 0,
            report.mvi_metrics.fill_rate * 100 if report.mvi_metrics.enabled else 0
        ]

        fig1.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Scores'
        ))

        fig1.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100])
            ),
            title="Quality Metrics Overview",
            showlegend=False
        )

        figures.append(fig1)

        # 2. Data Quality Bar Chart
        fig2 = go.Figure()

        fig2.add_trace(go.Bar(
            x=['Completeness', 'Consistency', 'Accuracy'],
            y=[
                report.data_quality.completeness_score * 100,
                report.data_quality.consistency_score * 100,
                report.data_quality.accuracy_score * 100
            ],
            marker_color=['#667eea', '#764ba2', '#f093fb']
        ))

        fig2.update_layout(
            title="Data Quality Dimensions",
            yaxis_title="Score (%)",
            yaxis_range=[0, 100]
        )

        figures.append(fig2)

        # 3. Column Quality Heatmap
        if report.data_quality.column_quality:
            cols = list(report.data_quality.column_quality.keys())[:20]  # Top 20
            null_rates = [report.data_quality.column_quality[c]["null_rate"] for c in cols]
            unique_rates = [report.data_quality.column_quality[c]["unique_rate"] for c in cols]

            fig3 = go.Figure()

            fig3.add_trace(go.Bar(
                x=cols,
                y=null_rates,
                name='Null Rate',
                marker_color='#e74c3c'
            ))

            fig3.add_trace(go.Bar(
                x=cols,
                y=unique_rates,
                name='Unique Rate',
                marker_color='#3498db'
            ))

            fig3.update_layout(
                title="Column Quality Profile (Top 20)",
                xaxis_title="Column",
                yaxis_title="Rate",
                barmode='group',
                xaxis_tickangle=-45
            )

            figures.append(fig3)

        # 4. Merge Quality Sankey
        fig4 = go.Figure(go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=["Input Rows", "Retained", "Lost", "Output"],
                color=["#667eea", "#2ecc71", "#e74c3c", "#764ba2"]
            ),
            link=dict(
                source=[0, 0, 1],
                target=[1, 2, 3],
                value=[
                    report.merge_quality.rows_retained,
                    report.merge_quality.rows_lost,
                    report.merge_quality.rows_retained
                ]
            )
        ))

        fig4.update_layout(
            title="Data Flow: Merge Process",
            font_size=12
        )

        figures.append(fig4)

        # 5. ER/MVI Performance (if enabled)
        if report.er_metrics.enabled or report.mvi_metrics.enabled:
            fig5 = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Entity Resolution", "Missing Value Imputation"),
                specs=[[{"type": "indicator"}, {"type": "indicator"}]]
            )

            if report.er_metrics.enabled:
                fig5.add_trace(
                    go.Indicator(
                        mode="gauge+number+delta",
                        value=report.er_metrics.deduplication_rate * 100,
                        title={'text': "Dedup Rate (%)"},
                        delta={'reference': 10},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "#667eea"},
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ),
                    row=1, col=1
                )

            if report.mvi_metrics.enabled:
                fig5.add_trace(
                    go.Indicator(
                        mode="gauge+number+delta",
                        value=report.mvi_metrics.fill_rate * 100,
                        title={'text': "Fill Rate (%)"},
                        delta={'reference': 20},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "#764ba2"},
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 80
                            }
                        }
                    ),
                    row=1, col=2
                )

            fig5.update_layout(title="Enhancement Effectiveness")
            figures.append(fig5)

        return figures


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="VLDB-Grade Validation Report Generator"
    )
    parser.add_argument(
        "--csv",
        required=True,
        type=Path,
        help="Path to merged CSV"
    )
    parser.add_argument(
        "--meta",
        type=Path,
        help="Path to metadata JSON (defaults to <csv>.meta.json)"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("validation_report"),
        help="Output path prefix (without extension)"
    )
    parser.add_argument(
        "--format",
        choices=["csv", "html", "both"],
        default="both",
        help="Output format"
    )

    args = parser.parse_args()

    # Resolve meta path
    if args.meta is None:
        args.meta = args.csv.with_suffix(".meta.json")

    # Validate inputs
    if not args.csv.exists():
        print(f"[ERROR] CSV not found: {args.csv}")
        return 1

    if not args.meta.exists():
        print(f"[ERROR] Metadata not found: {args.meta}")
        return 1

    # Generate report
    print(f"[VLDB Validator] Processing {args.csv.name}...")

    try:
        engine = ValidationEngine(args.csv, args.meta)
        report = engine.generate_report()

        # Export
        if args.format in ("csv", "both"):
            csv_out = args.out.with_suffix(".csv")
            CSVExporter.export(report, csv_out)

        if args.format in ("html", "both"):
            html_out = args.out.with_suffix(".html")
            HTMLExporter.export(report, html_out)

        print(f"\n[SUCCESS] Overall Quality Score: {report.overall_score}/100")
        print(f"[SUCCESS] Reports generated at: {args.out}.*")

        return 0

    except Exception as e:
        print(f"[ERROR] Validation failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())