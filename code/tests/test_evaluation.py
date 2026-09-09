"""
tests/test_evaluation.py

Tests for code/evaluation/main.py.

Each test targets a specific error class that the framework is designed
to detect. Tests construct minimal in-memory CSV payloads — no dataset
files are required.

Run from the repo root:
    python -m pytest code/tests/test_evaluation.py -v
"""

from __future__ import annotations

import csv
import io
import os
import sys
import tempfile
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from evaluation.main import EvaluationFramework, EvaluationReport


# ── fixtures and helpers ──────────────────────────────────────────────────────

MESSAGES_HEADER = [
    "message_id", "user_id", "conversation_type", "group_id", "business_id",
    "sender_user_id", "created_at", "message_text", "media_type",
    "media_id", "forwarded_count",
]

HISTORY_HEADER = [
    "message_id", "user_id", "sender_id", "message_text", "event_type",
    "event_timestamp",
]

OUTPUT_HEADER = [
    "message_id", "action", "message_type", "reason", "confidence",
    "evidence_message_ids",
]

SAMPLE_HEADER = [
    "message_id", "action", "message_type", "reason", "confidence",
    "evidence_message_ids",
]


def _write_csv(rows: list[list[Any]], header: list[str]) -> str:
    """Write rows to a temp file; return file path."""
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8"
    )
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)
    f.close()
    return f.name


def _make_messages(n: int = 5) -> str:
    rows = [
        [f"msg_{i:03d}", "u_001", "personal", "", "", f"u_00{i}",
         "2026-07-01 10:00", "Hello", "", "", "0"]
        for i in range(1, n + 1)
    ]
    return _write_csv(rows, MESSAGES_HEADER)


def _make_history(n: int = 10) -> str:
    rows = [
        [f"message_{i:04d}", "u_001", f"u_00{i % 5 + 1}",
         "some text", "opened", "2026-06-01 09:00"]
        for i in range(1, n + 1)
    ]
    return _write_csv(rows, HISTORY_HEADER)


def _make_output(
    n: int = 5,
    action: str = "digest",
    message_type: str = "personal",
    reason: str = "The message is safe but non-urgent and can be reviewed later.",
    confidence: str = "0.78",
    evidence: str = "none",
    ids: list[str] | None = None,
) -> str:
    ids = ids or [f"msg_{i:03d}" for i in range(1, n + 1)]
    rows = [
        [mid, action, message_type, reason, confidence, evidence]
        for mid in ids
    ]
    return _write_csv(rows, OUTPUT_HEADER)


def _evaluate(
    messages_path: str,
    history_path: str,
    output_path: str,
    *,
    dataset_dir: str | None = None,
) -> EvaluationReport:
    """Convenience wrapper that builds a temp dataset_dir."""
    if dataset_dir is None:
        # Build a minimal dataset_dir from the paths given
        d = tempfile.mkdtemp()
        import shutil
        shutil.copy(messages_path, os.path.join(d, "messages.csv"))
        shutil.copy(history_path, os.path.join(d, "message_history.csv"))
        dataset_dir = d

    return EvaluationFramework().evaluate(output_path, dataset_dir)


# ── C1: Completeness ──────────────────────────────────────────────────────────

class TestCompleteness:

    def test_pass_when_all_ids_present(self, tmp_path):
        msgs  = _make_messages(5)
        hist  = _make_history(5)
        out   = _make_output(5, ids=[f"msg_{i:03d}" for i in range(1, 6)])
        report = _evaluate(msgs, hist, out)
        c1 = next(c for c in report.checks if "Completeness" in c.name)
        assert c1.status == "PASS"

    def test_fail_on_missing_row(self, tmp_path):
        msgs  = _make_messages(5)
        hist  = _make_history(5)
        # Only 4 rows — msg_005 is missing
        out   = _make_output(4, ids=[f"msg_{i:03d}" for i in range(1, 5)])
        report = _evaluate(msgs, hist, out)
        c1 = next(c for c in report.checks if "Completeness" in c.name)
        assert c1.status == "FAIL"
        assert any("missing" in issue.lower() for issue in c1.issues)

    def test_fail_on_duplicate_ids(self, tmp_path):
        msgs  = _make_messages(5)
        hist  = _make_history(5)
        # msg_001 appears twice
        ids = ["msg_001", "msg_001", "msg_002", "msg_003", "msg_004"]
        out   = _make_output(5, ids=ids)
        report = _evaluate(msgs, hist, out)
        c1 = next(c for c in report.checks if "Completeness" in c.name)
        assert c1.status == "FAIL"
        assert any("duplicate" in issue.lower() for issue in c1.issues)

    def test_warn_on_extra_ids(self, tmp_path):
        msgs  = _make_messages(5)
        hist  = _make_history(5)
        # 5 correct + 1 fabricated
        ids = [f"msg_{i:03d}" for i in range(1, 6)] + ["msg_999"]
        out   = _make_output(6, ids=ids)
        report = _evaluate(msgs, hist, out)
        c1 = next(c for c in report.checks if "Completeness" in c.name)
        assert c1.status == "WARN"

    def test_detail_shows_expected_and_actual_counts(self, tmp_path):
        msgs  = _make_messages(10)
        hist  = _make_history(5)
        out   = _make_output(10, ids=[f"msg_{i:03d}" for i in range(1, 11)])
        report = _evaluate(msgs, hist, out)
        c1 = next(c for c in report.checks if "Completeness" in c.name)
        assert "expected_rows" in c1.details
        assert c1.details["expected_rows"] == 10


# ── C2: Schema Correctness ────────────────────────────────────────────────────

class TestSchema:

    def test_pass_on_valid_output(self, tmp_path):
        msgs   = _make_messages(3)
        hist   = _make_history(5)
        out    = _make_output(3, ids=["msg_001", "msg_002", "msg_003"])
        report = _evaluate(msgs, hist, out)
        c2 = next(c for c in report.checks if "Schema" in c.name)
        assert c2.status == "PASS"

    def test_fail_on_invalid_action(self, tmp_path):
        msgs = _make_messages(3)
        hist = _make_history(5)
        out  = _make_output(3, action="DIGEST",  # wrong case
                            ids=["msg_001", "msg_002", "msg_003"])
        report = _evaluate(msgs, hist, out)
        c2 = next(c for c in report.checks if "Schema" in c.name)
        assert c2.status == "FAIL"
        assert any("action" in issue.lower() for issue in c2.issues)

    def test_fail_on_invalid_message_type(self, tmp_path):
        msgs = _make_messages(3)
        hist = _make_history(5)
        out  = _make_output(3, message_type="chat",  # not in vocabulary
                            ids=["msg_001", "msg_002", "msg_003"])
        report = _evaluate(msgs, hist, out)
        c2 = next(c for c in report.checks if "Schema" in c.name)
        assert c2.status == "FAIL"
        assert any("message_type" in issue.lower() for issue in c2.issues)

    def test_fail_on_confidence_as_percentage(self, tmp_path):
        msgs = _make_messages(3)
        hist = _make_history(5)
        out  = _make_output(3, confidence="91",  # should be 0.91
                            ids=["msg_001", "msg_002", "msg_003"])
        report = _evaluate(msgs, hist, out)
        c2 = next(c for c in report.checks if "Schema" in c.name)
        assert c2.status == "FAIL"
        assert any("confidence" in issue.lower() for issue in c2.issues)

    def test_fail_on_non_numeric_confidence(self, tmp_path):
        msgs = _make_messages(3)
        hist = _make_history(5)
        out  = _make_output(3, confidence="high",
                            ids=["msg_001", "msg_002", "msg_003"])
        report = _evaluate(msgs, hist, out)
        c2 = next(c for c in report.checks if "Schema" in c.name)
        assert c2.status == "FAIL"

    def test_fail_on_empty_evidence_field(self, tmp_path):
        msgs = _make_messages(3)
        hist = _make_history(5)
        out  = _make_output(3, evidence="",  # should be 'none' at minimum
                            ids=["msg_001", "msg_002", "msg_003"])
        report = _evaluate(msgs, hist, out)
        c2 = next(c for c in report.checks if "Schema" in c.name)
        assert c2.status == "FAIL"

    def test_all_valid_message_types_accepted(self, tmp_path):
        valid_types = [
            "urgent", "personal", "event", "payment", "business_update",
            "promotion", "greeting", "forward", "spam", "scam", "unknown",
        ]
        msgs = _make_messages(len(valid_types))
        hist = _make_history(5)
        ids  = [f"msg_{i:03d}" for i in range(1, len(valid_types) + 1)]

        # Build output with different type per row
        rows = [
            [mid, "digest", mtype,
             "The message is safe but non-urgent and can wait.",
             "0.78", "none"]
            for mid, mtype in zip(ids, valid_types)
        ]
        out_path = _write_csv(rows, OUTPUT_HEADER)

        d = tempfile.mkdtemp()
        import shutil
        shutil.copy(msgs, os.path.join(d, "messages.csv"))
        shutil.copy(hist, os.path.join(d, "message_history.csv"))
        report = EvaluationFramework().evaluate(out_path, d)

        c2 = next(c for c in report.checks if "Schema" in c.name)
        assert c2.status == "PASS", c2.issues

    def test_fail_on_missing_column(self, tmp_path):
        msgs = _make_messages(3)
        hist = _make_history(5)

        # Write output with missing column
        incomplete_header = ["message_id", "action", "message_type", "reason"]
        rows = [["msg_001", "digest", "personal", "reason text"]]
        out = _write_csv(rows, incomplete_header)

        d = tempfile.mkdtemp()
        import shutil
        shutil.copy(msgs, os.path.join(d, "messages.csv"))
        shutil.copy(hist, os.path.join(d, "message_history.csv"))
        report = EvaluationFramework().evaluate(out, d)

        c2 = next(c for c in report.checks if "Schema" in c.name)
        assert c2.status == "FAIL"
        assert any("missing" in i.lower() for i in c2.issues)


# ── C3: Action Distribution ───────────────────────────────────────────────────

class TestActionDistribution:

    def test_pass_on_balanced_distribution(self, tmp_path):
        msgs = _make_messages(9)
        hist = _make_history(5)
        ids  = [f"msg_{i:03d}" for i in range(1, 10)]
        rows = [
            [ids[0], "notify", "urgent",   "A critical message from a trusted sender.", "0.87", "none"],
            [ids[1], "notify", "personal",  "A close contact sent an urgent request.", "0.85", "none"],
            [ids[2], "notify", "event",     "A school event update from an admin.", "0.83", "none"],
            [ids[3], "digest", "personal",  "Safe casual chat with no urgent action.", "0.78", "none"],
            [ids[4], "digest", "promotion", "Relevant offer but does not need attention.", "0.78", "none"],
            [ids[5], "digest", "business_update", "A legitimate update from a verified business.", "0.78", "none"],
            [ids[6], "mute",   "scam",      "Phishing message from an unknown sender.", "0.85", "none"],
            [ids[7], "mute",   "forward",   "Viral chain forward with no urgency.", "0.82", "none"],
            [ids[8], "mute",   "greeting",  "Repeated greetings the user usually ignores.", "0.80", "none"],
        ]
        out = _write_csv(rows, OUTPUT_HEADER)

        d = tempfile.mkdtemp()
        import shutil
        shutil.copy(msgs, os.path.join(d, "messages.csv"))
        shutil.copy(hist, os.path.join(d, "message_history.csv"))
        report = EvaluationFramework().evaluate(out, d)

        c3 = next(c for c in report.checks if "Distribution" in c.name)
        assert c3.status == "PASS", c3.issues

    def test_fail_when_one_action_dominates(self, tmp_path):
        msgs = _make_messages(10)
        hist = _make_history(5)
        # 100% digest — clearly a bug
        out  = _make_output(10, action="digest",
                            ids=[f"msg_{i:03d}" for i in range(1, 11)])
        report = _evaluate(msgs, hist, out)
        c3 = next(c for c in report.checks if "Distribution" in c.name)
        assert c3.status == "FAIL"

    def test_warn_when_action_missing(self, tmp_path):
        msgs = _make_messages(6)
        hist = _make_history(5)
        ids  = [f"msg_{i:03d}" for i in range(1, 7)]
        # No mute action at all
        rows = [
            [ids[0], "notify", "urgent",   "Critical message from trusted sender.", "0.87", "none"],
            [ids[1], "notify", "personal", "Close contact urgent request.", "0.85", "none"],
            [ids[2], "notify", "event",    "Admin event update.", "0.83", "none"],
            [ids[3], "digest", "personal", "Safe casual chat.", "0.78", "none"],
            [ids[4], "digest", "promotion","Relevant offer.", "0.78", "none"],
            [ids[5], "digest", "business_update", "Verified update.", "0.78", "none"],
        ]
        out = _write_csv(rows, OUTPUT_HEADER)

        d = tempfile.mkdtemp()
        import shutil
        shutil.copy(msgs, os.path.join(d, "messages.csv"))
        shutil.copy(hist, os.path.join(d, "message_history.csv"))
        report = EvaluationFramework().evaluate(out, d)

        c3 = next(c for c in report.checks if "Distribution" in c.name)
        # Mute is 0% — should at least warn (scam content exists in dataset)
        assert c3.status in ("WARN", "FAIL")

    def test_details_show_counts_and_fractions(self, tmp_path):
        msgs = _make_messages(6)
        hist = _make_history(5)
        ids  = [f"msg_{i:03d}" for i in range(1, 7)]
        rows = [
            [ids[0], "notify", "urgent",   "Critical.", "0.87", "none"],
            [ids[1], "notify", "personal", "Close contact.", "0.85", "none"],
            [ids[2], "digest", "personal", "Casual chat.", "0.78", "none"],
            [ids[3], "digest", "promotion","Relevant.", "0.78", "none"],
            [ids[4], "mute",   "scam",     "Phishing.", "0.85", "none"],
            [ids[5], "mute",   "forward",  "Chain forward.", "0.82", "none"],
        ]
        out = _write_csv(rows, OUTPUT_HEADER)

        d = tempfile.mkdtemp()
        import shutil
        shutil.copy(msgs, os.path.join(d, "messages.csv"))
        shutil.copy(hist, os.path.join(d, "message_history.csv"))
        report = EvaluationFramework().evaluate(out, d)

        c3 = next(c for c in report.checks if "Distribution" in c.name)
        assert "notify" in c3.details
        assert "digest" in c3.details
        assert "mute" in c3.details
        assert "2" in c3.details["notify"]  # 2 notify messages


# ── C4: Evidence ID Correctness ───────────────────────────────────────────────

class TestEvidenceIds:

    def test_pass_on_none_evidence(self, tmp_path):
        msgs = _make_messages(3)
        hist = _make_history(5)
        out  = _make_output(3, evidence="none",
                            ids=["msg_001", "msg_002", "msg_003"])
        report = _evaluate(msgs, hist, out)
        c4 = next(c for c in report.checks if "Evidence" in c.name)
        assert c4.status == "PASS"

    def test_pass_on_valid_history_ids(self, tmp_path):
        msgs = _make_messages(3)
        hist = _make_history(5)  # contains message_0001 through message_0005

        rows = [
            ["msg_001", "notify", "urgent",
             "Trusted sender sent urgent message.", "0.87",
             "message_0001;message_0002"],
            ["msg_002", "digest", "personal",
             "Casual message from known contact.", "0.78", "none"],
            ["msg_003", "mute",   "scam",
             "Phishing from unknown sender.", "0.85", "none"],
        ]
        out = _write_csv(rows, OUTPUT_HEADER)

        d = tempfile.mkdtemp()
        import shutil
        shutil.copy(msgs, os.path.join(d, "messages.csv"))
        shutil.copy(hist, os.path.join(d, "message_history.csv"))
        report = EvaluationFramework().evaluate(out, d)

        c4 = next(c for c in report.checks if "Evidence" in c.name)
        assert c4.status == "PASS", c4.issues

    def test_fail_on_wrong_namespace_msg_prefix(self, tmp_path):
        msgs = _make_messages(3)
        hist = _make_history(5)

        # msg_001 prefix — from messages.csv, NOT history
        rows = [
            ["msg_001", "notify", "urgent",
             "Trusted sender.", "0.87", "msg_005"],  # wrong namespace!
            ["msg_002", "digest", "personal",
             "Casual.", "0.78", "none"],
            ["msg_003", "mute", "scam",
             "Phishing.", "0.85", "none"],
        ]
        out = _write_csv(rows, OUTPUT_HEADER)

        d = tempfile.mkdtemp()
        import shutil
        shutil.copy(msgs, os.path.join(d, "messages.csv"))
        shutil.copy(hist, os.path.join(d, "message_history.csv"))
        report = EvaluationFramework().evaluate(out, d)

        c4 = next(c for c in report.checks if "Evidence" in c.name)
        assert c4.status == "FAIL"
        assert any("wrong table" in issue.lower() for issue in c4.issues)

    def test_fail_on_wrong_namespace_sample_prefix(self, tmp_path):
        msgs = _make_messages(3)
        hist = _make_history(5)

        rows = [
            ["msg_001", "notify", "urgent",
             "Trusted sender.", "0.87", "sample_msg_001"],  # sample namespace!
            ["msg_002", "digest", "personal", "Casual.", "0.78", "none"],
            ["msg_003", "mute",   "scam",    "Phishing.", "0.85", "none"],
        ]
        out = _write_csv(rows, OUTPUT_HEADER)

        d = tempfile.mkdtemp()
        import shutil
        shutil.copy(msgs, os.path.join(d, "messages.csv"))
        shutil.copy(hist, os.path.join(d, "message_history.csv"))
        report = EvaluationFramework().evaluate(out, d)

        c4 = next(c for c in report.checks if "Evidence" in c.name)
        assert c4.status == "FAIL"

    def test_fail_on_nonexistent_history_id(self, tmp_path):
        msgs = _make_messages(3)
        hist = _make_history(5)  # only up to message_0005

        rows = [
            ["msg_001", "notify", "urgent",
             "Trusted sender.", "0.87", "message_9999"],  # does not exist
            ["msg_002", "digest", "personal", "Casual.", "0.78", "none"],
            ["msg_003", "mute",   "scam",    "Phishing.", "0.85", "none"],
        ]
        out = _write_csv(rows, OUTPUT_HEADER)

        d = tempfile.mkdtemp()
        import shutil
        shutil.copy(msgs, os.path.join(d, "messages.csv"))
        shutil.copy(hist, os.path.join(d, "message_history.csv"))
        report = EvaluationFramework().evaluate(out, d)

        c4 = next(c for c in report.checks if "Evidence" in c.name)
        assert c4.status == "FAIL"

    def test_warn_on_duplicate_evidence_ids_in_row(self, tmp_path):
        msgs = _make_messages(3)
        hist = _make_history(5)

        rows = [
            ["msg_001", "notify", "urgent",
             "Trusted sender.", "0.87", "message_0001;message_0001"],  # dup!
            ["msg_002", "digest", "personal", "Casual.", "0.78", "none"],
            ["msg_003", "mute",   "scam",    "Phishing.", "0.85", "none"],
        ]
        out = _write_csv(rows, OUTPUT_HEADER)

        d = tempfile.mkdtemp()
        import shutil
        shutil.copy(msgs, os.path.join(d, "messages.csv"))
        shutil.copy(hist, os.path.join(d, "message_history.csv"))
        report = EvaluationFramework().evaluate(out, d)

        c4 = next(c for c in report.checks if "Evidence" in c.name)
        assert c4.status in ("WARN", "FAIL")

    def test_details_show_rows_with_and_without_evidence(self, tmp_path):
        msgs = _make_messages(4)
        hist = _make_history(5)

        rows = [
            ["msg_001", "notify", "urgent",    "Trusted.", "0.87", "message_0001"],
            ["msg_002", "digest", "personal",  "Casual.",  "0.78", "none"],
            ["msg_003", "mute",   "scam",      "Scam.",    "0.85", "none"],
            ["msg_004", "digest", "business_update", "Update.", "0.78", "message_0002"],
        ]
        out = _write_csv(rows, OUTPUT_HEADER)

        d = tempfile.mkdtemp()
        import shutil
        shutil.copy(msgs, os.path.join(d, "messages.csv"))
        shutil.copy(hist, os.path.join(d, "message_history.csv"))
        report = EvaluationFramework().evaluate(out, d)

        c4 = next(c for c in report.checks if "Evidence" in c.name)
        assert c4.details["rows_with_evidence"] == 2
        assert c4.details["rows_no_evidence"] == 2


# ── C5: Output Quality (reason + DSI) ────────────────────────────────────────

class TestOutputQuality:

    def test_pass_on_good_quality_output(self, tmp_path):
        msgs = _make_messages(3)
        hist = _make_history(5)

        rows = [
            ["msg_001", "notify", "urgent",
             "A trusted group admin sent a time-sensitive update that should interrupt the user.",
             "0.87", "none"],
            ["msg_002", "digest", "personal",
             "The sender is trusted, but the message has no urgent action or safety relevance.",
             "0.78", "none"],
            ["msg_003", "mute",   "scam",
             "The message asks for urgent OTP or account verification through a suspicious flow.",
             "0.85", "none"],
        ]
        out = _write_csv(rows, OUTPUT_HEADER)

        d = tempfile.mkdtemp()
        import shutil
        shutil.copy(msgs, os.path.join(d, "messages.csv"))
        shutil.copy(hist, os.path.join(d, "message_history.csv"))
        report = EvaluationFramework().evaluate(out, d)

        c5 = next(c for c in report.checks if "Quality" in c.name)
        assert c5.status == "PASS", c5.issues

    def test_fail_on_all_same_confidence(self, tmp_path):
        msgs = _make_messages(5)
        hist = _make_history(5)

        # All the same confidence — zero variance = pipeline bug
        rows = [
            [f"msg_{i:03d}", "digest", "personal",
             "Safe message that can be reviewed later without urgency.",
             "0.78", "none"]  # all 0.78
            for i in range(1, 6)
        ]
        out = _write_csv(rows, OUTPUT_HEADER)

        d = tempfile.mkdtemp()
        import shutil
        shutil.copy(msgs, os.path.join(d, "messages.csv"))
        shutil.copy(hist, os.path.join(d, "message_history.csv"))
        report = EvaluationFramework().evaluate(out, d)

        c5 = next(c for c in report.checks if "Quality" in c.name)
        assert c5.status == "FAIL"
        assert any("identical confidence" in i.lower() or "zero" in i.lower()
                   for i in c5.issues)

    def test_fail_on_generic_reason(self, tmp_path):
        msgs = _make_messages(3)
        hist = _make_history(5)

        rows = [
            ["msg_001", "digest", "unknown", "none", "0.78", "none"],  # 'none' reason!
            ["msg_002", "digest", "personal",
             "Safe message with no urgent action required.", "0.79", "none"],
            ["msg_003", "mute", "scam",
             "Phishing attempt from unknown sender.", "0.85", "none"],
        ]
        out = _write_csv(rows, OUTPUT_HEADER)

        d = tempfile.mkdtemp()
        import shutil
        shutil.copy(msgs, os.path.join(d, "messages.csv"))
        shutil.copy(hist, os.path.join(d, "message_history.csv"))
        report = EvaluationFramework().evaluate(out, d)

        c5 = next(c for c in report.checks if "Quality" in c.name)
        assert c5.status == "FAIL"

    def test_warn_on_short_reason(self, tmp_path):
        msgs = _make_messages(3)
        hist = _make_history(5)

        rows = [
            ["msg_001", "digest", "personal", "OK", "0.78", "none"],  # too short
            ["msg_002", "digest", "personal",
             "Safe message with no urgent action required.", "0.79", "none"],
            ["msg_003", "mute",   "scam",
             "Phishing attempt from unknown sender.", "0.85", "none"],
        ]
        out = _write_csv(rows, OUTPUT_HEADER)

        d = tempfile.mkdtemp()
        import shutil
        shutil.copy(msgs, os.path.join(d, "messages.csv"))
        shutil.copy(hist, os.path.join(d, "message_history.csv"))
        report = EvaluationFramework().evaluate(out, d)

        c5 = next(c for c in report.checks if "Quality" in c.name)
        assert c5.status in ("WARN", "FAIL")

    def test_warn_on_hardcoded_confidence_050(self, tmp_path):
        msgs = _make_messages(3)
        hist = _make_history(5)

        rows = [
            ["msg_001", "notify", "urgent",
             "A trusted admin sent a time-sensitive update.", "0.50", "none"],
            ["msg_002", "digest", "personal",
             "Casual chat from a known contact.", "0.51", "none"],
            ["msg_003", "mute",   "scam",
             "Phishing attempt from unknown sender.", "0.52", "none"],
        ]
        out = _write_csv(rows, OUTPUT_HEADER)

        d = tempfile.mkdtemp()
        import shutil
        shutil.copy(msgs, os.path.join(d, "messages.csv"))
        shutil.copy(hist, os.path.join(d, "message_history.csv"))
        report = EvaluationFramework().evaluate(out, d)

        c5 = next(c for c in report.checks if "Quality" in c.name)
        # 0.50 is a suspect hardcoded value, and all are below DSI floor
        assert c5.status in ("WARN", "FAIL")

    def test_details_contain_confidence_stats(self, tmp_path):
        msgs = _make_messages(3)
        hist = _make_history(5)

        rows = [
            ["msg_001", "notify", "urgent",
             "A trusted admin sent a time-sensitive update.", "0.87", "none"],
            ["msg_002", "digest", "personal",
             "Casual chat from a known contact.", "0.78", "none"],
            ["msg_003", "mute",   "scam",
             "Phishing attempt from unknown sender.", "0.85", "none"],
        ]
        out = _write_csv(rows, OUTPUT_HEADER)

        d = tempfile.mkdtemp()
        import shutil
        shutil.copy(msgs, os.path.join(d, "messages.csv"))
        shutil.copy(hist, os.path.join(d, "message_history.csv"))
        report = EvaluationFramework().evaluate(out, d)

        c5 = next(c for c in report.checks if "Quality" in c.name)
        assert "confidence_min" in c5.details
        assert "confidence_max" in c5.details
        assert "confidence_unique_values" in c5.details
        assert c5.details["confidence_unique_values"] == 3


# ── Overall report ────────────────────────────────────────────────────────────

class TestOverallReport:

    def test_overall_fail_when_any_check_fails(self, tmp_path):
        msgs = _make_messages(3)
        hist = _make_history(5)
        # Missing a row → C1 FAIL
        out  = _make_output(2, ids=["msg_001", "msg_002"])
        report = _evaluate(msgs, hist, out)
        assert report.overall_status == "FAIL"
        assert report.passed is False

    def test_overall_pass_on_clean_output(self, tmp_path):
        msgs = _make_messages(3)
        hist = _make_history(5)

        rows = [
            ["msg_001", "notify", "urgent",
             "A trusted admin sent a time-sensitive update that needs attention.", "0.87", "none"],
            ["msg_002", "digest", "personal",
             "Casual chat from a known contact with no urgency.", "0.78", "none"],
            ["msg_003", "mute",   "scam",
             "Phishing attempt from an unknown sender asking for OTP.", "0.85", "none"],
        ]
        out = _write_csv(rows, OUTPUT_HEADER)

        d = tempfile.mkdtemp()
        import shutil
        shutil.copy(msgs, os.path.join(d, "messages.csv"))
        shutil.copy(hist, os.path.join(d, "message_history.csv"))
        report = EvaluationFramework().evaluate(out, d)

        # With only 3 messages, distribution warnings are expected (< 5% floor)
        # But no hard failures should occur
        assert report.passed

    def test_report_to_dict_is_serialisable(self, tmp_path):
        import json
        msgs = _make_messages(3)
        hist = _make_history(5)
        out  = _make_output(3, ids=["msg_001", "msg_002", "msg_003"])
        report = _evaluate(msgs, hist, out)
        d = report.to_dict()
        # Should serialise to JSON without error
        serialised = json.dumps(d)
        assert isinstance(serialised, str)
        assert "overall_status" in d

    def test_file_not_found_returns_fail_report(self, tmp_path):
        d = tempfile.mkdtemp()
        msgs_path = os.path.join(d, "messages.csv")
        with open(msgs_path, "w") as f:
            f.write("message_id\nmsg_001\n")
        report = EvaluationFramework().evaluate(
            "/nonexistent/output.csv", d
        )
        assert report.overall_status == "FAIL"
        assert any("not found" in i.lower() for c in report.checks for i in c.issues)


# ── C6: Sample accuracy ───────────────────────────────────────────────────────

class TestSampleAccuracy:

    def _make_sample(self, rows: list[list]) -> str:
        return _write_csv(rows, SAMPLE_HEADER)

    def _make_sample_predictions(self, rows: list[list]) -> str:
        return _write_csv(rows, OUTPUT_HEADER)

    def test_pass_on_perfect_accuracy(self, tmp_path):
        sample = self._make_sample([
            ["sample_msg_001", "notify", "urgent",
             "Trusted sender.", "0.87", "none"],
            ["sample_msg_002", "mute",   "scam",
             "Phishing.", "0.85", "none"],
        ])
        preds = self._make_sample_predictions([
            ["sample_msg_001", "notify", "urgent",
             "Trusted sender.", "0.87", "none"],
            ["sample_msg_002", "mute",   "scam",
             "Phishing.", "0.85", "none"],
        ])

        # Use minimal dataset_dir
        msgs = _make_messages(1)
        hist = _make_history(1)
        out  = _make_output(1, ids=["msg_001"])

        d = tempfile.mkdtemp()
        import shutil
        shutil.copy(msgs, os.path.join(d, "messages.csv"))
        shutil.copy(hist, os.path.join(d, "message_history.csv"))
        shutil.copy(sample, os.path.join(d, "sample_messages.csv"))

        report = EvaluationFramework().evaluate(out, d, sample_predictions_csv=preds)
        c6 = next((c for c in report.checks if "Sample" in c.name), None)
        assert c6 is not None
        assert c6.status == "PASS"
        assert c6.details.get("action_accuracy") == "100.0%"

    def test_warn_on_low_accuracy(self, tmp_path):
        sample = self._make_sample([
            [f"sample_msg_{i:03d}", "notify", "urgent",
             "Trusted urgent.", "0.87", "none"]
            for i in range(1, 11)
        ])
        preds = self._make_sample_predictions([
            [f"sample_msg_{i:03d}", "mute", "scam",
             "Phishing.", "0.85", "none"]  # all wrong
            for i in range(1, 11)
        ])

        msgs = _make_messages(1)
        hist = _make_history(1)
        out  = _make_output(1, ids=["msg_001"])

        d = tempfile.mkdtemp()
        import shutil
        shutil.copy(msgs, os.path.join(d, "messages.csv"))
        shutil.copy(hist, os.path.join(d, "message_history.csv"))
        shutil.copy(sample, os.path.join(d, "sample_messages.csv"))

        report = EvaluationFramework().evaluate(out, d, sample_predictions_csv=preds)
        c6 = next((c for c in report.checks if "Sample" in c.name), None)
        assert c6 is not None
        assert c6.status == "WARN"
        assert any("60%" in i or "accuracy" in i.lower() for i in c6.issues)

    def test_warn_when_no_sample_predictions_provided(self, tmp_path):
        msgs = _make_messages(3)
        hist = _make_history(5)
        out  = _make_output(3, ids=["msg_001", "msg_002", "msg_003"])

        sample_rows = [
            ["sample_msg_001", "notify", "urgent",
             "Trusted urgent.", "0.87", "none"],
        ]
        sample = self._make_sample(sample_rows)

        d = tempfile.mkdtemp()
        import shutil
        shutil.copy(msgs, os.path.join(d, "messages.csv"))
        shutil.copy(hist, os.path.join(d, "message_history.csv"))
        shutil.copy(sample, os.path.join(d, "sample_messages.csv"))
        shutil.copy(out, os.path.join(d, "output.csv"))

        report = EvaluationFramework().evaluate(
            os.path.join(d, "output.csv"),
            d,
            sample_predictions_csv=None,  # no predictions provided
        )
        c6 = next((c for c in report.checks if "Sample" in c.name), None)
        assert c6 is not None
        assert c6.status == "WARN"
