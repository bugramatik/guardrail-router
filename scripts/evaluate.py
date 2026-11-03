"""Run the screening pipeline over the labelled sample set and report quality.

Usage:
    python scripts/evaluate.py --file data/samples.jsonl

The script prints precision, recall and F1 for the block decision, the stage mix
and the latency percentiles, and exits non-zero when recall drops below the
regression gate so it can be used in CI.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guardrail_router.classifier import PromptClassifier  # noqa: E402
from guardrail_router.config import Settings  # noqa: E402
from guardrail_router.pipeline import ScreeningPipeline  # noqa: E402
from guardrail_router.slm import SlmJudge  # noqa: E402

RECALL_GATE = 0.85


def load_samples(path: Path) -> list[dict]:
    samples: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if "text" not in record or record.get("label") not in {"benign", "injection", "exfiltration"}:
                raise ValueError(f"{path}:{line_no}: missing text or unknown label")
            samples.append(record)
    return samples


def build_pipeline(settings: Settings) -> ScreeningPipeline:
    classifier = PromptClassifier(settings.model_path)
    judge = SlmJudge(settings.slm_endpoint, settings.slm_timeout_ms)
    return ScreeningPipeline(classifier, judge, settings)


async def evaluate(samples: list[dict], pipeline: ScreeningPipeline) -> dict:
    true_positive = false_positive = false_negative = true_negative = 0
    stages: dict[str, int] = {}
    latencies: list[float] = []
    per_category: dict[str, dict[str, int]] = {}

    for sample in samples:
        verdict = await pipeline.screen(sample["text"], sample.get("source"))
        latencies.append(verdict.latency_ms)
        stages[verdict.stage] = stages.get(verdict.stage, 0) + 1
        expected_block = sample["label"] != "benign"
        blocked = verdict.decision == "block"
        if expected_block and blocked:
            true_positive += 1
        elif expected_block and not blocked:
            false_negative += 1
        elif blocked:
            false_positive += 1
        else:
            true_negative += 1
        if expected_block:
            bucket = per_category.setdefault(sample["label"], {"total": 0, "caught": 0})
            bucket["total"] += 1
            bucket["caught"] += int(blocked)

    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = _ratio(2 * precision * recall, precision + recall)
    latencies.sort()
    return {
        "samples": len(samples),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
        },
        "stages": stages,
        "latency_ms": {
            "p50": round(_percentile(latencies, 0.50), 2),
            "p95": round(_percentile(latencies, 0.95), 2),
            "p99": round(_percentile(latencies, 0.99), 2),
        },
        "per_category": per_category,
    }


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(int(round(q * (len(sorted_values) - 1))), len(sorted_values) - 1)
    return sorted_values[index]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the guardrail screening pipeline")
    parser.add_argument("--file", type=Path, default=ROOT / "data" / "samples.jsonl")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)

    samples = load_samples(args.file)
    settings = Settings()
    pipeline = build_pipeline(settings)
    report = asyncio.run(evaluate(samples, pipeline))

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"samples      {report['samples']}")
        print(f"precision    {report['precision']:.3f}")
        print(f"recall       {report['recall']:.3f}")
        print(f"f1           {report['f1']:.3f}")
        print(f"stages       {report['stages']}")
        print(f"latency ms   {report['latency_ms']}")
        for label, counts in sorted(report["per_category"].items()):
            print(f"  {label:<12} {counts['caught']}/{counts['total']} caught")

    if report["recall"] < RECALL_GATE:
        print(f"recall {report['recall']:.3f} below gate {RECALL_GATE:.2f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
