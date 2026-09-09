"""
evaluation/main.py

Standalone evaluation framework for the Message Notification Router.

Validates output.csv against five correctness dimensions and optionally
compares against sample ground truth.

Usage (from repo root):
    python -m evaluation.main                        # uses defaults
    python -m evaluation.main --output dataset/output.csv --dataset dataset/
    python -m evaluation.main --sample-predictions dataset/sample_predictions.csv

    # With explicit paths:
    PYTHONPATH=code python code/evaluation/main.py

Exit code:
    0  PASS or WARN (submission is submittable)
    1  FAIL         (submission has structural errors; do not submit)

────────────────────────────────────────────────────────────────────────
HOW THE FRAMEWORK PREVENTS COMMON IMPLEMENTATION ERRORS

Error class                    | Check that catches it
───────────────────────────────┼───────────────────────────────────────
Missing rows in output         | C1 — completeness (every msg_id present)
Extra / duplicate rows         | C1 — duplicates detected
Wrong column order / name      | C2 — exact column header check
Invalid action value           | C2 — action ∈ {notify, digest, mute}
Invalid message_type value     | C2 — type ∈ vocabulary
Confidence out of [0,1]        | C2 — numeric range check
All messages get same action   | C3 — degenerate distribution detection
All mute / all notify          | C3 — warns if any action < 5%
Evidence ID not in history     | C4 — cross-references history.csv IDs
Evidence ID from wrong table   | C4 — rejects msg_xxx / sample_msg_xxx
Malformed evidence ID string   | C4 — semicolon-split validation
Reason field empty / too long  | C5 — length [40, 250] enforced
Confidence same for all rows   | C5 — zero-variance detection
Confidence < our DSI floor     | C5 — warns if < 0.55
────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal, Optional

# ── constants ─────────────────────────────────────────────────────────────────

REQUIRED_COLUMNS = (
    "message_id",
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
)

VALID_ACTIONS: frozenset[str] = frozenset({"notify", "digest", "mute"})

# Derived from sample_messages.csv ground truth — the only valid vocabulary
VALID_MESSAGE_TYPES: frozenset[str] = frozenset({
    "urgent",
    "personal",
    "event",
    "payment",
    "business_update",
    "promotion",
    "greeting",
    "forward",
    "spam",
    "scam",
    "unknown",
})

# Minimum reason length (chars) — shorter strings are likely truncated or empty
REASON_MIN_LENGTH = 30
REASON_MAX_LENGTH = 300

# DSI floor — our formula clamps to [0.60, 0.97]
DSI_WARN_FLOOR = 0.55

# Distribution: warn if any action is below this fraction of total messages
ACTION_MIN_FRACTION = 0.05


# ── result types ──────────────────────────────────────────────────────────────

CheckStatus = Literal["PASS", "WARN", "FAIL"]


@dataclass
class EvaluationCheck:
    """Result of one validation dimension."""

    name: str
    status: CheckStatus
    issues: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def _emoji(self) -> str:
        return {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}[self.status]

    def summary_line(self) -> str:
        return f"  {self._emoji()} [{self.status:<4}] {self.name}"

    def detail_lines(self) -> list[str]:
        lines = []
        for issue in self.issues:
            lines.append(f"           → {issue}")
        for k, v in self.details.items():
            lines.append(f"           {k}: {v}")
        return lines


@dataclass
class EvaluationReport:
    """Aggregate result of all evaluation checks."""

    checks: list[EvaluationCheck] = field(default_factory=list)

    @property
    def overall_status(self) -> CheckStatus:
        if any(c.status == "FAIL" for c in self.checks):
            return "FAIL"
        if any(c.status == "WARN" for c in self.checks):
            return "WARN"
        return "PASS"

    @property
    def passed(self) -> bool:
        """True if no FAIL checks — submission is structurally valid."""
        return self.overall_status != "FAIL"

    def print_report(self, verbose: bool = False) -> None:
        """Print a human-readable report to stdout."""
        overall = self.overall_status
        bar = "═" * 60
        emoji_map = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}

        print(bar)
        print(f" Evaluation Report  [{emoji_map[overall]} {overall}]")
        print(bar)

        for check in self.checks:
            print(check.summary_line())
            if verbose or check.status != "PASS":
                for line in check.detail_lines():
                    print(line)

        print(bar)

        fail_count = sum(1 for c in self.checks if c.status == "FAIL")
        warn_count = sum(1 for c in self.checks if c.status == "WARN")
        pass_count = sum(1 for c in self.checks if c.status == "PASS")
        print(f" {pass_count} passed  {warn_count} warnings  {fail_count} failed")
        print(bar)

        if overall == "FAIL":
            print(" ✗ DO NOT SUBMIT — structural errors must be fixed first.")
        elif overall == "WARN":
            print(" ⚠ Warnings detected — review before submitting.")
        else:
            print(" ✓ Output is structurally valid and ready to submit.")
        print(bar)

    def to_dict(self) -> dict:
        return {
            "overall_status": self.overall_status,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status,
                    "issues": c.issues,
                    "details": c.details,
                }
                for c in self.checks
            ],
        }


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _read_csv(path: str) -> list[dict]:
    """Read CSV, stripping BOM and whitespace from headers."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return [
            {k.strip(): (v.strip() if v else "") for k, v in row.items()}
            for row in reader
        ]


# ── EvaluationFramework ───────────────────────────────────────────────────────

class EvaluationFramework:
    """
    Validates output.csv against five correctness dimensions.

    Call evaluate() once. The returned EvaluationReport is immutable
    and can be printed or serialised.
    """

    def evaluate(
        self,
        output_csv: str,
        dataset_dir: str,
        sample_predictions_csv: Optional[str] = None,
    ) -> EvaluationReport:
        """
        Run all checks and return an EvaluationReport.

        Parameters
        ----------
        output_csv              path to the submitted output.csv
        dataset_dir             path to the dataset/ directory
        sample_predictions_csv  optional: predictions on sample_messages.csv,
                                used for accuracy estimation
        """
        report = EvaluationReport()

        # ── load output ───────────────────────────────────────────────────
        try:
            output_rows = _read_csv(output_csv)
        except FileNotFoundError:
            report.checks.append(EvaluationCheck(
                name="C0 — Output File Exists",
                status="FAIL",
                issues=[f"output.csv not found at: {output_csv}"],
            ))
            return report
        except Exception as exc:
            report.checks.append(EvaluationCheck(
                name="C0 — Output File Readable",
                status="FAIL",
                issues=[f"Failed to parse output.csv: {exc}"],
            ))
            return report

        # ── load dataset reference files ───────────────────────────────────
        messages_path = os.path.join(dataset_dir, "messages.csv")
        history_path  = os.path.join(dataset_dir, "message_history.csv")
        sample_path   = os.path.join(dataset_dir, "sample_messages.csv")

        try:
            messages_rows = _read_csv(messages_path)
        except FileNotFoundError:
            report.checks.append(EvaluationCheck(
                name="C0 — Dataset Accessible",
                status="FAIL",
                issues=[f"messages.csv not found at: {messages_path}"],
            ))
            return report

        history_rows = []
        if os.path.exists(history_path):
            history_rows = _read_csv(history_path)

        sample_rows: list[dict] = []
        if os.path.exists(sample_path):
            sample_rows = _read_csv(sample_path)

        expected_ids: set[str] = {r["message_id"] for r in messages_rows}
        history_ids: set[str] = {r["message_id"] for r in history_rows}

        # ── run checks ────────────────────────────────────────────────────
        report.checks.append(
            self._check_completeness(output_rows, expected_ids)
        )
        report.checks.append(
            self._check_schema(output_rows)
        )
        report.checks.append(
            self._check_action_distribution(output_rows)
        )
        report.checks.append(
            self._check_evidence_ids(output_rows, history_ids)
        )
        report.checks.append(
            self._check_output_quality(output_rows)
        )

        # ── optional sample comparison ─────────────────────────────────────
        if sample_predictions_csv and os.path.exists(sample_predictions_csv):
            sample_pred_rows = _read_csv(sample_predictions_csv)
            if sample_rows:
                report.checks.append(
                    self._check_sample_accuracy(sample_pred_rows, sample_rows)
                )
        elif sample_rows:
            # If no sample predictions given, report that calibration is available
            report.checks.append(EvaluationCheck(
                name="C6 — Sample Accuracy (calibration)",
                status="WARN",
                issues=[
                    "sample_messages.csv exists but no sample predictions were provided.",
                    "Run the system on sample_messages.csv and pass --sample-predictions "
                    "to measure accuracy.",
                ],
            ))

        return report

    # ------------------------------------------------------------------
    # C1 — Completeness
    # ------------------------------------------------------------------

    def _check_completeness(
        self,
        output_rows: list[dict],
        expected_ids: set[str],
    ) -> EvaluationCheck:
        """
        Every message_id in messages.csv must appear exactly once in output.csv.

        Catches:
        - Missing predictions (pipeline silently skipped a message)
        - Duplicate predictions (pipeline processed a message twice)
        - Predictions for messages not in the input (ID fabrication bug)
        """
        found_ids: list[str] = [r.get("message_id", "").strip() for r in output_rows]
        found_set = set(found_ids)

        issues = []
        details: dict = {}

        missing = sorted(expected_ids - found_set)
        extra   = sorted(found_set - expected_ids - {""})

        # Duplicates
        counts = Counter(found_ids)
        duplicates = sorted(k for k, v in counts.items() if v > 1)

        details["expected_rows"] = len(expected_ids)
        details["actual_rows"]   = len(output_rows)

        if missing:
            issues.append(f"{len(missing)} message(s) missing: {missing[:5]}{'…' if len(missing) > 5 else ''}")
        if extra:
            issues.append(f"{len(extra)} unexpected ID(s) in output: {extra[:5]}")
        if duplicates:
            issues.append(f"{len(duplicates)} duplicate message_id(s): {duplicates[:5]}")

        if "" in found_set:
            issues.append("One or more rows have an empty message_id")

        status: CheckStatus = (
            "FAIL" if (missing or duplicates or "" in found_set)
            else "WARN" if extra
            else "PASS"
        )

        return EvaluationCheck(
            name="C1 — Completeness (one row per message)",
            status=status,
            issues=issues,
            details=details,
        )

    # ------------------------------------------------------------------
    # C2 — Schema correctness
    # ------------------------------------------------------------------

    def _check_schema(self, output_rows: list[dict]) -> EvaluationCheck:
        """
        Validates every field in every row against the output contract:
        - Correct column names present
        - action ∈ {notify, digest, mute}
        - message_type ∈ vocabulary
        - confidence is a float in (0, 1]
        - reason is a non-empty string
        - evidence_message_ids is 'none' or semicolon-separated

        Catches:
        - Typos in column names ('Action' vs 'action')
        - Enum drift ('NOTIFY' instead of 'notify')
        - Confidence stored as percentage (91 instead of 0.91)
        - Missing or malformed evidence_id field
        """
        issues: list[str] = []

        if not output_rows:
            return EvaluationCheck(
                name="C2 — Schema Correctness",
                status="FAIL",
                issues=["output.csv is empty"],
            )

        # Check column names
        actual_columns = set(output_rows[0].keys())
        required_set   = set(REQUIRED_COLUMNS)
        missing_cols   = required_set - actual_columns
        if missing_cols:
            issues.append(f"Missing columns: {sorted(missing_cols)}")
            return EvaluationCheck(
                name="C2 — Schema Correctness",
                status="FAIL",
                issues=issues,
            )

        action_errors:  list[str] = []
        type_errors:    list[str] = []
        conf_errors:    list[str] = []
        evidence_errors: list[str] = []

        for row in output_rows:
            msg_id = row.get("message_id", "?")

            # action
            action = row.get("action", "").strip()
            if action not in VALID_ACTIONS:
                action_errors.append(f"{msg_id}: action={action!r}")

            # message_type
            mtype = row.get("message_type", "").strip()
            if mtype not in VALID_MESSAGE_TYPES:
                type_errors.append(f"{msg_id}: message_type={mtype!r}")

            # confidence
            raw_conf = row.get("confidence", "").strip()
            try:
                conf = float(raw_conf)
                if not (0.0 < conf <= 1.0):
                    conf_errors.append(f"{msg_id}: confidence={conf} (must be in (0, 1])")
            except ValueError:
                conf_errors.append(f"{msg_id}: confidence={raw_conf!r} (not a number)")

            # evidence_message_ids — basic format check
            ev_field = row.get("evidence_message_ids", "").strip()
            if ev_field == "":
                evidence_errors.append(f"{msg_id}: evidence_message_ids is empty (use 'none')")

        def _cap(lst: list[str], cap: int = 5) -> list[str]:
            if len(lst) > cap:
                return lst[:cap] + [f"… and {len(lst) - cap} more"]
            return lst

        if action_errors:
            issues.append(f"Invalid action values ({len(action_errors)} rows):")
            issues.extend(_cap(action_errors))

        if type_errors:
            issues.append(f"Invalid message_type values ({len(type_errors)} rows):")
            issues.extend(_cap(type_errors))

        if conf_errors:
            issues.append(f"Invalid confidence values ({len(conf_errors)} rows):")
            issues.extend(_cap(conf_errors))

        if evidence_errors:
            issues.append(f"Invalid evidence_message_ids ({len(evidence_errors)} rows):")
            issues.extend(_cap(evidence_errors))

        status: CheckStatus = "FAIL" if (
            action_errors or type_errors or conf_errors or evidence_errors
        ) else "PASS"

        return EvaluationCheck(
            name="C2 — Schema Correctness (values and types)",
            status=status,
            issues=issues,
            details={
                "rows_checked": len(output_rows),
                "action_errors": len(action_errors),
                "type_errors": len(type_errors),
                "confidence_errors": len(conf_errors),
                "evidence_format_errors": len(evidence_errors),
            },
        )

    # ------------------------------------------------------------------
    # C3 — Action distribution
    # ------------------------------------------------------------------

    def _check_action_distribution(
        self, output_rows: list[dict]
    ) -> EvaluationCheck:
        """
        Validates that the action distribution is not degenerate.

        A system that routes 100% of messages to 'digest' is not a router —
        it's a bug. The sample ground truth shows ~30% each. We warn when
        any action is below 5% of total.

        Also warns when 'mute' fraction is suspiciously low, because the
        dataset contains known scam and spam messages.

        Catches:
        - Pipeline always returning 'digest' (fallback leak)
        - Policy engine never triggering 'notify'
        - Inference engine over-flagging everything as SCAM → all mute
        """
        issues: list[str] = []
        total = len(output_rows)

        if total == 0:
            return EvaluationCheck(
                name="C3 — Action Distribution",
                status="FAIL",
                issues=["No rows to evaluate"],
            )

        counts = Counter(r.get("action", "").strip() for r in output_rows)
        distribution = {
            a: round(counts.get(a, 0) / total, 3) for a in VALID_ACTIONS
        }

        for action, frac in distribution.items():
            if frac < ACTION_MIN_FRACTION:
                issues.append(
                    f"Action '{action}' is {frac:.1%} of output "
                    f"({counts.get(action, 0)}/{total}) — may indicate a bug"
                )

        # If mute is < 10%, warn specifically: dataset has known scam content
        mute_frac = distribution.get("mute", 0.0)
        if mute_frac < 0.10:
            issues.append(
                f"'mute' fraction ({mute_frac:.1%}) is unexpectedly low — "
                "the dataset contains known scam/spam content that should be muted"
            )

        # If any single action > 80%, it's almost certainly a bug
        for action, frac in distribution.items():
            if frac > 0.80:
                issues.append(
                    f"Action '{action}' dominates at {frac:.1%} — "
                    "system may not be differentiating messages"
                )

        status: CheckStatus = (
            "FAIL" if any(v > 0.80 for v in distribution.values())
            else "WARN" if issues
            else "PASS"
        )

        return EvaluationCheck(
            name="C3 — Action Distribution (non-degenerate)",
            status=status,
            issues=issues,
            details={
                "total_messages": total,
                "notify":  f"{counts.get('notify', 0)} ({distribution.get('notify', 0):.1%})",
                "digest":  f"{counts.get('digest', 0)} ({distribution.get('digest', 0):.1%})",
                "mute":    f"{counts.get('mute', 0)} ({distribution.get('mute', 0):.1%})",
            },
        )

    # ------------------------------------------------------------------
    # C4 — Evidence retrieval correctness
    # ------------------------------------------------------------------

    def _check_evidence_ids(
        self,
        output_rows: list[dict],
        history_ids: set[str],
    ) -> EvaluationCheck:
        """
        Every evidence_message_id must either be 'none' or exist in
        message_history.csv.

        Namespace rules (derived from dataset ID prefixes):
          - Valid:   'none'
          - Valid:   'message_NNNN' (from message_history.csv)
          - Invalid: 'msg_NNN'       (from messages.csv — wrong table)
          - Invalid: 'sample_msg_NNN'(from sample_messages.csv — wrong table)
          - Invalid: anything else

        Catches:
        - Historical match returning IDs from the wrong table
        - Pipeline inventing message IDs not in the dataset
        - Off-by-one in ID formatting ('message_001' vs 'message_0001')
        - Duplicate evidence IDs within one row (redundant evidence)
        """
        issues: list[str] = []
        not_in_history: list[str] = []
        wrong_namespace: list[str] = []
        duplicates_found: list[str] = []

        for row in output_rows:
            msg_id  = row.get("message_id", "?")
            ev_raw  = row.get("evidence_message_ids", "none").strip()

            if ev_raw.lower() == "none" or ev_raw == "":
                continue

            ev_ids = [e.strip() for e in ev_raw.split(";") if e.strip()]

            # Detect duplicates within this row
            seen: set[str] = set()
            for eid in ev_ids:
                if eid in seen:
                    duplicates_found.append(f"{msg_id}: duplicate '{eid}'")
                seen.add(eid)

            for eid in ev_ids:
                # Wrong namespace checks
                if eid.startswith("msg_") or eid.startswith("sample_msg_"):
                    wrong_namespace.append(f"{msg_id}: '{eid}' is from wrong table")
                elif eid not in history_ids and not eid.startswith("message_"):
                    not_in_history.append(f"{msg_id}: unknown ID '{eid}'")
                elif eid not in history_ids:
                    not_in_history.append(f"{msg_id}: '{eid}' not found in message_history.csv")

        def _cap(lst: list[str], cap: int = 5) -> list[str]:
            return lst[:cap] + ([f"… and {len(lst) - cap} more"] if len(lst) > cap else [])

        if not_in_history:
            issues.append(f"Evidence IDs not found in message_history.csv ({len(not_in_history)}):")
            issues.extend(_cap(not_in_history))

        if wrong_namespace:
            issues.append(f"Evidence IDs from wrong table ({len(wrong_namespace)}):")
            issues.extend(_cap(wrong_namespace))

        if duplicates_found:
            issues.append(f"Duplicate evidence IDs in single row ({len(duplicates_found)}):")
            issues.extend(_cap(duplicates_found))

        if not history_ids:
            issues.append(
                "message_history.csv could not be loaded — evidence ID validation skipped"
            )
            status: CheckStatus = "WARN"
        else:
            status = (
                "FAIL" if (not_in_history or wrong_namespace)
                else "WARN" if duplicates_found
                else "PASS"
            )

        rows_with_evidence = sum(
            1 for r in output_rows
            if r.get("evidence_message_ids", "none").strip().lower() != "none"
        )

        return EvaluationCheck(
            name="C4 — Evidence ID Correctness",
            status=status,
            issues=issues,
            details={
                "rows_with_evidence": rows_with_evidence,
                "rows_no_evidence": len(output_rows) - rows_with_evidence,
                "invalid_ids": len(not_in_history) + len(wrong_namespace),
            },
        )

    # ------------------------------------------------------------------
    # C5 — Output quality (DSI + reason)
    # ------------------------------------------------------------------

    def _check_output_quality(self, output_rows: list[dict]) -> EvaluationCheck:
        """
        Validates the quality of the `reason` and `confidence` fields.

        For reason:
        - Must be between REASON_MIN_LENGTH and REASON_MAX_LENGTH characters
        - Must not be a generic placeholder ('unknown', 'N/A', etc.)

        For confidence:
        - Must not be the same value for ALL rows (zero-variance = pipeline bug)
        - Values below DSI_WARN_FLOOR are suspicious (our formula floors at 0.60)
        - Values of exactly 0.5 suggest a hardcoded fallback

        Catches:
        - Reason field left as empty string or 'none'
        - DSI computation always returning the same number
        - All messages getting confidence=0.5 (hardcoded fallback)
        - Reason strings that are way too long (LLM hallucination leak)
        """
        issues: list[str] = []

        reason_too_short: list[str] = []
        reason_too_long:  list[str] = []
        reason_generic:   list[str] = []
        conf_below_floor: list[str] = []
        conf_hardcoded:   list[str] = []

        GENERIC_PLACEHOLDERS = {"unknown", "n/a", "none", "null", "", "reason", "no reason"}
        HARDCODED_SUSPECT = {0.5, 0.50, 1.0}  # common fallback values

        confidences: list[float] = []

        for row in output_rows:
            msg_id = row.get("message_id", "?")
            reason = row.get("reason", "").strip()

            if len(reason) < REASON_MIN_LENGTH:
                reason_too_short.append(f"{msg_id}: len={len(reason)} reason={reason[:40]!r}")
            if len(reason) > REASON_MAX_LENGTH:
                reason_too_long.append(f"{msg_id}: len={len(reason)}")
            if reason.lower() in GENERIC_PLACEHOLDERS:
                reason_generic.append(f"{msg_id}: reason={reason!r}")

            raw_conf = row.get("confidence", "").strip()
            try:
                conf = float(raw_conf)
                confidences.append(conf)
                if conf < DSI_WARN_FLOOR:
                    conf_below_floor.append(f"{msg_id}: confidence={conf:.3f}")
                if conf in HARDCODED_SUSPECT:
                    conf_hardcoded.append(f"{msg_id}: confidence={conf} (suspect hardcoded value)")
            except ValueError:
                pass  # already caught in C2

        def _cap(lst: list[str], cap: int = 5) -> list[str]:
            return lst[:cap] + ([f"… and {len(lst) - cap} more"] if len(lst) > cap else [])

        # Zero-variance confidence
        if confidences and len(set(confidences)) == 1:
            issues.append(
                f"All {len(confidences)} rows have identical confidence={confidences[0]} "
                "— likely a hardcoded fallback"
            )

        if reason_too_short:
            issues.append(f"Reason too short < {REASON_MIN_LENGTH} chars ({len(reason_too_short)} rows):")
            issues.extend(_cap(reason_too_short))

        if reason_too_long:
            issues.append(f"Reason too long > {REASON_MAX_LENGTH} chars ({len(reason_too_long)} rows):")
            issues.extend(_cap(reason_too_long))

        if reason_generic:
            issues.append(f"Reason is a generic placeholder ({len(reason_generic)} rows):")
            issues.extend(_cap(reason_generic))

        if conf_below_floor:
            issues.append(
                f"Confidence below DSI floor {DSI_WARN_FLOOR} ({len(conf_below_floor)} rows):"
            )
            issues.extend(_cap(conf_below_floor))

        if conf_hardcoded:
            issues.append(f"Suspect hardcoded confidence values ({len(conf_hardcoded)} rows):")
            issues.extend(_cap(conf_hardcoded))

        # Compute summary stats
        details: dict = {}
        if confidences:
            details["confidence_min"]  = f"{min(confidences):.3f}"
            details["confidence_max"]  = f"{max(confidences):.3f}"
            details["confidence_mean"] = f"{sum(confidences)/len(confidences):.3f}"
            details["confidence_unique_values"] = len(set(confidences))

        is_fail = bool(
            reason_generic
            or (len(set(confidences)) == 1 and len(confidences) > 1)
        )
        is_warn = bool(
            reason_too_short or reason_too_long
            or conf_below_floor or conf_hardcoded
        )

        status: CheckStatus = "FAIL" if is_fail else "WARN" if is_warn else "PASS"

        return EvaluationCheck(
            name="C5 — Output Quality (reason + DSI validity)",
            status=status,
            issues=issues,
            details=details,
        )

    # ------------------------------------------------------------------
    # C6 — Sample accuracy (optional calibration check)
    # ------------------------------------------------------------------

    def _check_sample_accuracy(
        self,
        pred_rows: list[dict],
        sample_rows: list[dict],
    ) -> EvaluationCheck:
        """
        Compare predictions on sample_messages.csv against ground truth.

        Reports:
        - Overall action accuracy (exact match)
        - Per-action precision / recall
        - Message type accuracy
        - Confusion matrix (as text)

        This check is WARN-level — we don't FAIL on low accuracy (the system
        might have legitimate disagreements with the sample labels).
        """
        issues: list[str] = []

        sample_gt: dict[str, dict] = {r["message_id"]: r for r in sample_rows}
        pred_map:  dict[str, dict] = {r["message_id"]: r for r in pred_rows}

        matched_ids = set(sample_gt.keys()) & set(pred_map.keys())
        if not matched_ids:
            return EvaluationCheck(
                name="C6 — Sample Accuracy (calibration)",
                status="WARN",
                issues=[
                    "No overlapping message_ids between sample predictions and ground truth.",
                    "Ensure your sample predictions file uses the same message_id format "
                    "as sample_messages.csv (e.g., sample_msg_001).",
                ],
            )

        # Action accuracy
        action_correct = sum(
            1 for mid in matched_ids
            if pred_map[mid].get("action") == sample_gt[mid].get("action")
        )
        action_accuracy = action_correct / len(matched_ids)

        # Type accuracy
        type_correct = sum(
            1 for mid in matched_ids
            if pred_map[mid].get("message_type") == sample_gt[mid].get("message_type")
        )
        type_accuracy = type_correct / len(matched_ids)

        # Per-action breakdown
        per_action: dict[str, dict[str, int]] = {
            a: {"tp": 0, "fp": 0, "fn": 0} for a in VALID_ACTIONS
        }
        for mid in matched_ids:
            pred_action = pred_map[mid].get("action", "")
            true_action = sample_gt[mid].get("action", "")
            if pred_action == true_action:
                per_action[true_action]["tp"] += 1
            else:
                per_action.get(pred_action, {})
                if pred_action in per_action:
                    per_action[pred_action]["fp"] += 1
                if true_action in per_action:
                    per_action[true_action]["fn"] += 1

        details: dict = {
            "matched_samples": len(matched_ids),
            "action_accuracy": f"{action_accuracy:.1%}",
            "type_accuracy":   f"{type_accuracy:.1%}",
        }

        for action, counts in per_action.items():
            tp = counts["tp"]
            fp = counts["fp"]
            fn = counts["fn"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            details[f"{action}_precision"] = f"{precision:.1%}"
            details[f"{action}_recall"]    = f"{recall:.1%}"

        if action_accuracy < 0.60:
            issues.append(
                f"Action accuracy {action_accuracy:.1%} is below 60% on sample — "
                "review policy engine decisions"
            )
        if type_accuracy < 0.50:
            issues.append(
                f"Message type accuracy {type_accuracy:.1%} is below 50% on sample — "
                "LLM classifier or type resolution may need tuning"
            )

        return EvaluationCheck(
            name="C6 — Sample Accuracy (calibration)",
            status="WARN" if issues else "PASS",
            issues=issues,
            details=details,
        )


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the notification router output.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m evaluation.main
  python -m evaluation.main --output dataset/output.csv --dataset dataset/
  python -m evaluation.main --sample-predictions dataset/sample_predictions.csv -v
""",
    )
    parser.add_argument(
        "--output", "-o",
        default="dataset/output.csv",
        help="Path to the system's output CSV (default: dataset/output.csv)",
    )
    parser.add_argument(
        "--dataset", "-d",
        default="dataset",
        help="Path to the dataset directory (default: dataset/)",
    )
    parser.add_argument(
        "--sample-predictions", "-s",
        default=None,
        dest="sample_predictions",
        help="Path to predictions on sample_messages.csv (optional, for accuracy check)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show all check details, not just failures and warnings",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print report as JSON instead of human-readable text",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns exit code (0 = pass/warn, 1 = fail)."""
    args = _parse_args(argv)

    framework = EvaluationFramework()
    report = framework.evaluate(
        output_csv=args.output,
        dataset_dir=args.dataset,
        sample_predictions_csv=args.sample_predictions,
    )

    if args.json:
        import json
        print(json.dumps(report.to_dict(), indent=2))
    else:
        report.print_report(verbose=args.verbose)

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
