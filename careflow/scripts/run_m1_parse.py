"""M1 batch runner — parse all 200 intake records and measure accuracy.

Run from the careflow/ project root:
    python scripts/run_m1_parse.py

Reads:  data/intake/records.jsonl  (200 records)
Writes: output/m1_parsed.jsonl     (one ParseResult per record)
        output/m1_accuracy.json    (summary statistics)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.patient_request import ParseResult, parse_intake_record

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("m1_runner")

DATA_DIR   = Path("data/intake/records.jsonl")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

PARSED_FILE   = OUTPUT_DIR / "m1_parsed.jsonl"
ACCURACY_FILE = OUTPUT_DIR / "m1_accuracy.json"

# ── Load records ───────────────────────────────────────────────────────────────
def load_records() -> list[dict]:
    records = []
    with open(DATA_DIR, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── Accuracy scoring ───────────────────────────────────────────────────────────
def score_result(result: ParseResult, ground_truth: dict) -> dict:
    """Compare parsed fields vs ground_truth block."""
    if not result.success or result.patient_request is None:
        return {"urgency_match": False, "clinical_match": False, "specialty_match": False}

    pr   = result.patient_request
    gt   = ground_truth

    return {
        "urgency_match":   pr.urgency == gt.get("urgency"),
        "clinical_match":  pr.seeks_clinical_advice == gt.get("seeks_clinical_advice"),
        "specialty_match": (
            pr.specialty == gt.get("specialty")
            if gt.get("specialty") is not None
            else pr.specialty is None
        ),
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    records = load_records()
    logger.info("Loaded %d intake records from %s", len(records), DATA_DIR)

    results_out   = []
    scores        = []
    failed_ids    = []

    for i, record in enumerate(records, start=1):
        record_id = record.get("record_id", f"record_{i}")
        logger.info("[%d/%d] Parsing %s …", i, len(records), record_id)

        result = parse_intake_record(record, max_retries=2)

        ground_truth = record.get("ground_truth", {})
        score        = score_result(result, ground_truth)
        scores.append(score)

        if not result.success:
            failed_ids.append(record_id)

        # Serialize ParseResult for JSONL output
        out_row = {
            "record_id":     result.record_id,
            "success":       result.success,
            "attempts":      result.attempts,
            "parse_errors":  result.parse_errors,
            "patient_request": (
                result.patient_request.model_dump(mode="json")
                if result.patient_request
                else None
            ),
            "ground_truth":  ground_truth,
            "score":         score,
        }
        results_out.append(out_row)

        # Throttle: avoid Gemini rate-limit (1 req/sec on free tier is safe)
        time.sleep(1.0)

    # ── Write output ───────────────────────────────────────────────────────────
    with open(PARSED_FILE, "w", encoding="utf-8") as f:
        for row in results_out:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    logger.info("Parsed results saved to %s", PARSED_FILE)

    # ── Accuracy summary ───────────────────────────────────────────────────────
    total         = len(records)
    success_count = sum(1 for r in results_out if r["success"])
    urgency_acc   = sum(1 for s in scores if s["urgency_match"])
    clinical_acc  = sum(1 for s in scores if s["clinical_match"])
    specialty_acc = sum(1 for s in scores if s["specialty_match"])

    summary = {
        "total_records":       total,
        "parse_success":       success_count,
        "parse_success_pct":   round(success_count / total * 100, 1),
        "urgency_accuracy":    urgency_acc,
        "urgency_accuracy_pct":   round(urgency_acc / total * 100, 1),
        "clinical_advice_accuracy":    clinical_acc,
        "clinical_advice_accuracy_pct": round(clinical_acc / total * 100, 1),
        "specialty_accuracy":  specialty_acc,
        "specialty_accuracy_pct": round(specialty_acc / total * 100, 1),
        "failed_record_ids":   failed_ids,
    }

    with open(ACCURACY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ── Print report ───────────────────────────────────────────────────────────
    print("\n" + "=" * 56)
    print("  M1 Parse Run — Results")
    print("=" * 56)
    print(f"  Total records       : {total}")
    print(f"  Parse success       : {success_count}/{total}  ({summary['parse_success_pct']}%)")
    print(f"  Urgency accuracy    : {urgency_acc}/{total}  ({summary['urgency_accuracy_pct']}%)")
    print(f"  Clinical advice acc : {clinical_acc}/{total}  ({summary['clinical_advice_accuracy_pct']}%)")
    print(f"  Specialty accuracy  : {specialty_acc}/{total}  ({summary['specialty_accuracy_pct']}%)")
    print(f"  Failed IDs          : {failed_ids or 'none'}")
    print(f"\n  Full results : {PARSED_FILE}")
    print(f"  Accuracy     : {ACCURACY_FILE}")
    print("=" * 56 + "\n")

    # M1 plan targets
    if summary["parse_success_pct"] < 95:
        print(f"  ⚠  Parse success {summary['parse_success_pct']}% < 95% target")
    if summary["urgency_accuracy_pct"] < 75:
        print(f"  ⚠  Urgency accuracy {summary['urgency_accuracy_pct']}% < 75% target")
    if summary["clinical_advice_accuracy_pct"] < 70:
        print(f"  ⚠  Clinical advice detection {summary['clinical_advice_accuracy_pct']}% < 70% target")


if __name__ == "__main__":
    main()
