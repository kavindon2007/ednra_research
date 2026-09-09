"""
inference/states.py

Typed state enumerations and the InferredMessageState bundle.

The inference engine converts raw EvidenceBundle fields into five named,
ordered states. The policy engine operates ONLY on InferredMessageState —
it never reads raw evidence floats directly.

State design principles:
  - Each enum value maps to exactly one named condition.
  - Severity/trust orderings are expressed as integers via helper methods
    so the policy engine can use comparison operators instead of name checks.
  - InferredMessageState is frozen — once inferred, no code can mutate it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class GuardFired:
    name: str
    details: str



# ---------------------------------------------------------------------------
# Content Risk State
# ---------------------------------------------------------------------------

class ContentRiskState(Enum):
    """
    Characterises the safety of the message content itself.

    Ordered from safest to most dangerous. Use .severity() for numeric
    comparisons.
    """

    CLEAN      = "clean"       # No risk signals detected
    SUSPICIOUS = "suspicious"  # Mild, low-confidence risk signals
    SPAM       = "spam"        # Repetitive/chain content — no safety risk
    PHISHING   = "phishing"    # Medium-confidence credential/fraud attempt
    SCAM       = "scam"        # High-confidence financial fraud
    INJECTION  = "injection"   # Prompt injection in message text

    _SEVERITY = {  # type: ignore[misc]
        "clean": 0, "suspicious": 1, "spam": 2,
        "phishing": 3, "scam": 4, "injection": 5,
    }

    def severity(self) -> int:
        return ContentRiskState._SEVERITY.value[self.value]  # type: ignore[attr-defined]

    def __lt__(self, other: ContentRiskState) -> bool:
        return self.severity() < other.severity()

    def __le__(self, other: ContentRiskState) -> bool:
        return self.severity() <= other.severity()

    def __gt__(self, other: ContentRiskState) -> bool:
        return self.severity() > other.severity()

    def __ge__(self, other: ContentRiskState) -> bool:
        return self.severity() >= other.severity()


# Fix: make _SEVERITY accessible as a plain dict (enum members shadow class attrs)
_CONTENT_RISK_SEVERITY: dict[str, int] = {
    "clean": 0, "suspicious": 1, "spam": 2,
    "phishing": 3, "scam": 4, "injection": 5,
}


def _content_risk_severity(state: ContentRiskState) -> int:
    return _CONTENT_RISK_SEVERITY[state.value]


# Patch the method to use the module-level dict
ContentRiskState.severity = lambda self: _content_risk_severity(self)  # type: ignore[method-assign]
ContentRiskState.__lt__ = lambda self, other: self.severity() < other.severity()  # type: ignore[method-assign]
ContentRiskState.__le__ = lambda self, other: self.severity() <= other.severity()  # type: ignore[method-assign]
ContentRiskState.__gt__ = lambda self, other: self.severity() > other.severity()  # type: ignore[method-assign]
ContentRiskState.__ge__ = lambda self, other: self.severity() >= other.severity()  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Sender Credibility State
# ---------------------------------------------------------------------------

class SenderCredibilityState(Enum):
    """
    Characterises how trustworthy the sender is relative to this user's
    history and platform verification status.

    Ordered from least trusted to most. Use .trust_score() for numeric
    comparisons.
    """

    IMPERSONATOR     = "impersonator"      # Confirmed brand impersonation
    KNOWN_NEGATIVE   = "known_negative"    # User has muted/reported this sender
    UNKNOWN          = "unknown"           # First contact, no prior relationship
    KNOWN_NEUTRAL    = "known_neutral"     # Known sender, mixed/neutral engagement
    KNOWN_TRUSTED    = "known_trusted"     # User replies consistently — high engagement
    VERIFIED_TRUSTED = "verified_trusted"  # Verified business, no domain mismatch

    _SCORES = {  # type: ignore[misc]
        "impersonator": -2, "known_negative": -1, "unknown": 0,
        "known_neutral": 1, "known_trusted": 2, "verified_trusted": 3,
    }


_SENDER_SCORES: dict[str, int] = {
    "impersonator": -2, "known_negative": -1, "unknown": 0,
    "known_neutral": 1, "known_trusted": 2, "verified_trusted": 3,
}


def _sender_trust_score(state: SenderCredibilityState) -> int:
    return _SENDER_SCORES[state.value]


SenderCredibilityState.trust_score = lambda self: _sender_trust_score(self)  # type: ignore[method-assign]
SenderCredibilityState.__lt__ = lambda self, other: self.trust_score() < other.trust_score()  # type: ignore[method-assign]
SenderCredibilityState.__le__ = lambda self, other: self.trust_score() <= other.trust_score()  # type: ignore[method-assign]
SenderCredibilityState.__gt__ = lambda self, other: self.trust_score() > other.trust_score()  # type: ignore[method-assign]
SenderCredibilityState.__ge__ = lambda self, other: self.trust_score() >= other.trust_score()  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Message Intent State
# ---------------------------------------------------------------------------

class MessageIntentState(Enum):
    """
    Characterises what the message is trying to accomplish.

    Values are ordered from most disruptive (URGENT) to least (FORWARD_CHAIN)
    for documentation clarity, but the policy engine uses explicit name
    comparisons rather than ordinal comparisons.
    """

    URGENT        = "urgent"        # Requires immediate action
    OPERATIONAL   = "operational"   # Actionable but not time-critical
    INFORMATIONAL = "informational" # Useful, non-actionable
    SOCIAL        = "social"        # Casual human interaction
    PROMOTIONAL   = "promotional"   # Marketing content
    FORWARD_CHAIN = "forward_chain" # Viral chain / wellness forward
    AMBIGUOUS     = "ambiguous"     # Insufficient signal to classify


# ---------------------------------------------------------------------------
# User Receptivity State
# ---------------------------------------------------------------------------

class UserReceptivityState(Enum):
    """
    Characterises the user's current willingness to receive interruptions
    from this specific sender/group/business at this moment.

    ACTIVELY_AVOIDING is the most restrictive; RECEPTIVE is the default.
    """

    ACTIVELY_AVOIDING = "actively_avoiding"  # Muted group / reported sender
    OPTED_OUT         = "opted_out"          # Explicit promotional opt-out
    DND               = "dnd"                # Inside do-not-disturb window
    FATIGUED          = "fatigued"           # High recent dismiss rate
    RECEPTIVE         = "receptive"          # No suppression signals


# ---------------------------------------------------------------------------
# Contextual Urgency State
# ---------------------------------------------------------------------------

class ContextualUrgencyState(Enum):
    """
    Characterises how time-sensitive this message is, given both content
    signals and the credibility of the sender.

    CRITICAL can override DND and ACTIVELY_AVOIDING in the policy engine.
    """

    NONE     = "none"     # No urgency (chain forwards, greetings)
    LOW      = "low"      # Useful but can wait indefinitely
    MEDIUM   = "medium"   # Should be seen today (delivery, event, update)
    HIGH     = "high"     # Should interrupt if user is available
    CRITICAL = "critical" # Must interrupt (life/work-blocking emergency)

    _LEVELS = {  # type: ignore[misc]
        "none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4,
    }


_URGENCY_LEVELS: dict[str, int] = {
    "none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4,
}


def _urgency_level(state: ContextualUrgencyState) -> int:
    return _URGENCY_LEVELS[state.value]


ContextualUrgencyState.urgency_level = lambda self: _urgency_level(self)  # type: ignore[method-assign]
ContextualUrgencyState.__lt__ = lambda self, other: self.urgency_level() < other.urgency_level()  # type: ignore[method-assign]
ContextualUrgencyState.__le__ = lambda self, other: self.urgency_level() <= other.urgency_level()  # type: ignore[method-assign]
ContextualUrgencyState.__gt__ = lambda self, other: self.urgency_level() > other.urgency_level()  # type: ignore[method-assign]
ContextualUrgencyState.__ge__ = lambda self, other: self.urgency_level() >= other.urgency_level()  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Inferred Message State — the bundle the policy engine operates on
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InferredMessageState:
    """
    The five inferred states for one message, plus one trace string per state.

    Produced by InferenceEngine.infer(bundle). Consumed by PolicyEngine.decide().
    Frozen — no code may mutate this after construction.

    Trace strings explain WHY each state was assigned. The policy engine
    uses them to build the human-readable `reason` field in RoutingDecision.
    """

    # ── five inferred states ──────────────────────────────────────────────
    content_risk:       ContentRiskState
    sender_credibility: SenderCredibilityState
    message_intent:     MessageIntentState
    user_receptivity:   UserReceptivityState
    contextual_urgency: ContextualUrgencyState

    # ── one structured trace per state ────────────────────────────────────
    content_risk_trace:       GuardFired
    sender_credibility_trace: GuardFired
    message_intent_trace:     GuardFired
    user_receptivity_trace:   GuardFired
    contextual_urgency_trace: GuardFired

    # ── injection flag: True when injection was detected but from a trusted sender ──
    # When True, the policy engine strips the preamble and routes on actual content.
    # When False + content_risk == INJECTION, the policy engine mutes.
    injection_from_trusted: bool = False
