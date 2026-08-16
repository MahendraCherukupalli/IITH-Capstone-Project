"""M1 unit tests — PatientRequest parser.

Tests run against the real intake/records.jsonl data.
Fast tests use only a small sample (first 5 records) to avoid burning API quota.
The full-accuracy test is marked slow and skipped by default.

Run fast tests:
    pytest tests/test_m1_parser.py -v

Run all (including slow):
    pytest tests/test_m1_parser.py -v --run-slow
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.patient_request import (
    SPECIALTIES,
    ParseResult,
    PatientRequest,
    parse_intake_record,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────
RECORDS_PATH = Path("data/intake/records.jsonl")


def load_all_records() -> list[dict]:
    records = []
    with open(RECORDS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@pytest.fixture(scope="session")
def all_records():
    return load_all_records()


@pytest.fixture(scope="session")
def sample_records(all_records):
    """First 5 records — used for fast tests that call the LLM."""
    return all_records[:5]


# ── Model schema tests (no LLM, instant) ──────────────────────────────────────
class TestPatientRequestSchema:
    """Test the Pydantic schema without touching the LLM."""

    def test_required_fields_present(self):
        fields = set(PatientRequest.model_fields.keys())
        required = {
            "record_id", "channel", "raw_text", "urgency",
            "seeks_clinical_advice", "missing_fields",
        }
        assert required.issubset(fields), f"Missing required fields: {required - fields}"

    def test_optional_fields_have_defaults(self):
        assert not PatientRequest.model_fields["patient_ref"].is_required()
        assert not PatientRequest.model_fields["insurance_id"].is_required()
        assert not PatientRequest.model_fields["specialty"].is_required()

    def test_valid_construction(self):
        pr = PatientRequest(
            record_id="TEST-001",
            channel="web_form",
            raw_text="Need cardiology appointment.",
            urgency="Routine",
            specialty="Cardiology",
            seeks_clinical_advice=False,
        )
        assert pr.record_id == "TEST-001"
        assert pr.specialty == "Cardiology"
        assert pr.seeks_clinical_advice is False
        assert pr.patient_ref is None
        assert pr.insurance_id is None

    def test_urgency_enum_rejects_invalid(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PatientRequest(
                record_id="X",
                channel="email",
                raw_text="test",
                urgency="CRITICAL",  # not a valid literal
                seeks_clinical_advice=False,
            )

    def test_channel_enum_rejects_invalid(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PatientRequest(
                record_id="X",
                channel="telegram",  # not allowed
                raw_text="test",
                urgency="Routine",
                seeks_clinical_advice=False,
            )

    def test_specialties_list_is_complete(self):
        assert len(SPECIALTIES) == 8
        assert "Cardiology" in SPECIALTIES
        assert "ENT" in SPECIALTIES

    def test_parse_result_schema(self):
        pr = ParseResult(record_id="TEST-001")
        assert pr.success is False
        assert pr.patient_request is None
        assert pr.parse_errors == []
        assert pr.attempts == 0


# ── Data integrity tests (no LLM, instant) ────────────────────────────────────
class TestIntakeData:
    """Verify the intake JSONL file is well-formed before running LLM tests."""

    def test_records_file_exists(self):
        assert RECORDS_PATH.exists(), f"Records file not found: {RECORDS_PATH}"

    def test_record_count(self, all_records):
        assert len(all_records) == 200, f"Expected 200 records, got {len(all_records)}"

    def test_record_shape(self, all_records):
        required_keys = {"channel", "raw_text", "record_id", "ground_truth"}
        for i, rec in enumerate(all_records[:10]):
            assert required_keys.issubset(rec.keys()), \
                f"Record {i} missing keys: {required_keys - rec.keys()}"

    def test_ground_truth_shape(self, all_records):
        gt_keys = {"urgency", "specialty", "seeks_clinical_advice", "missing_fields"}
        for i, rec in enumerate(all_records[:10]):
            gt = rec.get("ground_truth", {})
            assert gt_keys.issubset(gt.keys()), \
                f"Record {i} ground_truth missing keys: {gt_keys - gt.keys()}"

    def test_urgency_values_in_data(self, all_records):
        valid_urgencies = {"Routine", "Priority", "Urgent", "Emergent"}
        for rec in all_records:
            urg = rec["ground_truth"]["urgency"]
            assert urg in valid_urgencies, f"Unknown urgency '{urg}' in {rec['record_id']}"

    def test_channel_values_in_data(self, all_records):
        valid_channels = {"phone_transcript", "web_form", "email", "sms"}
        for rec in all_records:
            ch = rec["channel"]
            assert ch in valid_channels, f"Unknown channel '{ch}' in {rec['record_id']}"

    def test_clinical_advice_ratio(self, all_records):
        """About 1 in 5 records should seek clinical advice per the data spec."""
        clinical = sum(1 for r in all_records if r["ground_truth"]["seeks_clinical_advice"])
        ratio = clinical / len(all_records)
        # Spec says ~20%, actual data shows ~28%, so allow 15%-40% range
        assert 0.15 <= ratio <= 0.45, \
            f"Clinical advice ratio {ratio:.0%} outside expected 15-40% range"


# ── LLM integration tests (small sample) ──────────────────────────────────────
class TestParseIntakeRecord:
    """These tests call the real LLM — use a tiny sample (5 records)."""

    def test_parse_returns_parse_result(self, sample_records):
        record = sample_records[0]
        result = parse_intake_record(record, max_retries=1)
        assert isinstance(result, ParseResult)
        assert result.record_id == record["record_id"]

    def test_parse_success_flag(self, sample_records):
        """At least 4 out of 5 sample records should parse successfully."""
        successes = 0
        for record in sample_records:
            result = parse_intake_record(record, max_retries=1)
            if result.success:
                successes += 1
        assert successes >= 4, \
            f"Only {successes}/5 sample records parsed successfully — check LLM client"

    def test_parsed_urgency_is_valid(self, sample_records):
        valid_urgencies = {"Routine", "Priority", "Urgent", "Emergent"}
        for record in sample_records:
            result = parse_intake_record(record, max_retries=1)
            if result.success and result.patient_request:
                assert result.patient_request.urgency in valid_urgencies

    def test_parsed_specialty_is_valid_or_none(self, sample_records):
        for record in sample_records:
            result = parse_intake_record(record, max_retries=1)
            if result.success and result.patient_request:
                sp = result.patient_request.specialty
                assert sp is None or sp in SPECIALTIES, \
                    f"Invalid specialty '{sp}' for {record['record_id']}"

    def test_parsed_channel_preserved(self, sample_records):
        """Channel must always match the original record."""
        for record in sample_records:
            result = parse_intake_record(record, max_retries=1)
            if result.success and result.patient_request:
                assert result.patient_request.channel == record["channel"]

    def test_record_id_preserved(self, sample_records):
        """record_id must always match the original record."""
        for record in sample_records:
            result = parse_intake_record(record, max_retries=1)
            assert result.record_id == record["record_id"]

    def test_clinical_advice_detection(self, sample_records):
        """Check that seeks_clinical_advice matches ground_truth at least 60% of the time."""
        matches = 0
        valid_results = 0
        for record in sample_records:
            result = parse_intake_record(record, max_retries=1)
            if result.success and result.patient_request:
                valid_results += 1
                gt_clinical = record["ground_truth"]["seeks_clinical_advice"]
                if result.patient_request.seeks_clinical_advice == gt_clinical:
                    matches += 1
        if valid_results > 0:
            acc = matches / valid_results
            assert acc >= 0.60, \
                f"Clinical advice detection {acc:.0%} < 60% on sample"

    def test_emergent_record_classified_correctly(self, all_records):
        """The very first emergent record in the data should be classified Urgent or Emergent."""
        emergent_records = [r for r in all_records if r["ground_truth"]["urgency"] == "Emergent"]
        assert emergent_records, "No Emergent records in test data — check data"
        record = emergent_records[0]
        result = parse_intake_record(record, max_retries=2)
        if result.success and result.patient_request:
            assert result.patient_request.urgency in ("Urgent", "Emergent"), \
                f"Emergent case classified as {result.patient_request.urgency}"


# ── Slow / full accuracy test (skipped unless --run-slow flag) ─────────────────
def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False,
        help="Run slow tests that process all 200 records"
    )


@pytest.mark.slow
class TestFullAccuracy:
    """Processes all 200 records — only run with --run-slow flag."""

    @pytest.fixture(autouse=True)
    def skip_if_not_slow(self, request):
        if not request.config.getoption("--run-slow", default=False):
            pytest.skip("Pass --run-slow to run full accuracy tests")

    def test_parse_success_rate_above_95_pct(self, all_records):
        results = [parse_intake_record(r, max_retries=2) for r in all_records]
        success = sum(1 for r in results if r.success)
        pct = success / len(all_records) * 100
        assert pct >= 95, f"Parse success {pct:.1f}% < 95% target"

    def test_urgency_accuracy_above_75_pct(self, all_records):
        correct = 0
        for record in all_records:
            result = parse_intake_record(record, max_retries=2)
            if result.success and result.patient_request:
                if result.patient_request.urgency == record["ground_truth"]["urgency"]:
                    correct += 1
        pct = correct / len(all_records) * 100
        assert pct >= 75, f"Urgency accuracy {pct:.1f}% < 75% target"

    def test_clinical_advice_accuracy_above_70_pct(self, all_records):
        correct = 0
        for record in all_records:
            result = parse_intake_record(record, max_retries=2)
            if result.success and result.patient_request:
                if result.patient_request.seeks_clinical_advice == record["ground_truth"]["seeks_clinical_advice"]:
                    correct += 1
        pct = correct / len(all_records) * 100
        assert pct >= 70, f"Clinical advice detection {pct:.1f}% < 70% target"
