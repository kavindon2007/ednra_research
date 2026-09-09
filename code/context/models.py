"""
context/models.py

All shared dataclasses for the notification routing system.
This is the single source of truth for every typed value that flows
through the pipeline. No module should define its own ad-hoc dicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Input data models — loaded from CSV files
# ---------------------------------------------------------------------------


@dataclass
class IncomingMessage:
    """One row from messages.csv — the unit of work for the entire system."""

    message_id: str
    user_id: str
    conversation_type: str          # personal | group | business
    group_id: Optional[str]
    business_id: Optional[str]
    sender_user_id: Optional[str]
    created_at: str                  # raw string; pipelines parse as needed
    message_text: str
    media_type: str                  # '' | 'image' | 'voice'
    media_id: Optional[str]
    forwarded_count: int


@dataclass
class UserProfile:
    """One row from users.csv."""

    user_id: str
    do_not_disturb_window: str       # e.g. "22:00-07:00"; empty if none
    messages_opened_30d: int
    messages_replied_30d: int
    notifications_dismissed_30d: int
    messages_reported_30d: int


@dataclass
class GroupInfo:
    """One row from groups.csv."""

    group_id: str
    group_name: str
    group_type: str                  # family | coworker | school_group | society | marketplace | …
    member_count: int
    admin_count: int
    created_at: str
    messages_30d: int


@dataclass
class GroupMembership:
    """One row from group_members.csv — per-user per-group relationship."""

    group_id: str
    user_id: str
    role: str                        # admin | member
    joined_at: str
    messages_sent_30d: int
    messages_read_30d: int
    replies_sent_30d: int
    notifications_dismissed_30d: int
    group_muted_by_user: bool


@dataclass
class BusinessInfo:
    """One row from business_accounts.csv."""

    business_id: str
    display_name: str
    brand_name: str
    category: str
    verified: bool
    official_domain: str
    domain_used_by_sender: str
    account_age_days: int
    messages_sent_30d: int
    user_reports_30d: int
    domain_used_by_sender_age_days: int


@dataclass
class UserBusinessRelation:
    """One row from user_business_history.csv — per-user per-business relationship."""

    user_id: str
    business_id: str
    why_user_knows_account: str      # e.g. active_bank_account, recent_grocery_delivery, …
    last_activity_at: str
    allows_promotions: bool
    promotions_opted_out_at: Optional[str]
    activity_count_180d: int
    messages_opened_30d: int
    messages_dismissed_30d: int
    messages_replied_30d: int
    last_reply_at: Optional[str]


@dataclass
class HistoricalMessage:
    """One row from message_history.csv — same schema as IncomingMessage."""

    message_id: str
    user_id: str
    conversation_type: str
    group_id: Optional[str]
    business_id: Optional[str]
    sender_user_id: Optional[str]
    created_at: str
    message_text: str
    media_type: str
    media_id: Optional[str]
    forwarded_count: int


@dataclass
class MessageEvent:
    """One row from message_events.csv — how the user reacted to a historical message."""

    user_id: str
    message_id: str
    message_opened: bool
    message_replied: bool
    reaction_time_minutes: Optional[int]   # None means not opened
    notification_dismissed: bool
    muted_after_message: bool
    message_reported: bool


@dataclass
class DailyNotificationSummary:
    """One row from daily_notification_summary.csv."""

    user_id: str
    date: str
    notifications_sent: int
    notifications_dismissed: int


@dataclass
class ImageRecord:
    """One row from images.csv."""

    image_id: str
    file_path: str


@dataclass
class VoiceNoteRecord:
    """One row from voice_notes.csv."""

    voice_note_id: str
    file_path: str


# ---------------------------------------------------------------------------
# Evidence models — output of each pipeline
# No pipeline emits a routing decision; only structured evidence.
# ---------------------------------------------------------------------------


@dataclass
class Evidence:
    """Base class for all pipeline evidence. Never used directly."""
    pass


@dataclass
class ScamRiskEvidence(Evidence):
    """
    From ScamRiskPipeline.
    risk_level: 'none' | 'low' | 'medium' | 'high'
    flags: human-readable reasons, e.g. ['domain_mismatch', 'otp_request']
    """

    risk_level: str
    flags: list[str] = field(default_factory=list)


@dataclass
class SenderTrustEvidence(Evidence):
    """
    From SenderTrustPipeline.
    trust_level: 'high' | 'medium' | 'low' | 'unknown'
    """

    trust_level: str
    is_admin: bool
    is_direct_mention: bool
    is_first_contact: bool
    sender_reply_rate: float          # fraction of past messages user replied to from this sender


@dataclass
class UserBehaviorEvidence(Evidence):
    """From UserBehaviorPipeline."""

    in_dnd: bool
    dismiss_rate: float               # notifications_dismissed / messages_opened (0–1)
    report_rate: float                # messages_reported / messages_opened (0–1)
    fatigue_score: float              # recent dismiss/sent ratio from daily summary (0–1)


@dataclass
class GroupContextEvidence(Evidence):
    """From GroupContextPipeline."""

    group_type: str                   # family | coworker | school_group | society | marketplace | unknown
    is_muted_by_user: bool
    user_engagement_ratio: float      # replies_sent / max(messages_read, 1)
    user_dismiss_rate_in_group: float # dismissed / max(messages_read, 1)
    sender_is_admin: bool


@dataclass
class BusinessContextEvidence(Evidence):
    """From BusinessContextPipeline."""

    is_verified: bool
    has_active_relation: bool         # user has meaningful history with this business
    opted_out: bool                   # user explicitly opted out of promotions
    user_open_ratio: float            # messages_opened / max(messages_opened + messages_dismissed, 1)
    relation_type: str                # e.g. active_bank_account, old_sale_subscription, ''


@dataclass
class ContentSignalsEvidence(Evidence):
    """From ContentSignalsPipeline — fully deterministic text heuristics."""

    urgency_score: float              # 0–1
    detected_patterns: list[str]      # e.g. ['otp_request', 'forward_chain', 'urgency_keyword']
    is_forward_chain: bool
    is_greeting_forward: bool
    has_payment_keyword: bool
    has_event_keyword: bool
    has_direct_mention: bool          # @user_id appears in text


@dataclass
class MediaSignalsEvidence(Evidence):
    """From MediaSignalsPipeline."""

    has_media: bool
    media_type: str                   # '' | 'image' | 'voice'
    media_id: str
    media_description: str            # filled in by LLM inference step; empty until then
    inferred_urgency: float           # 0–1; 0 until LLM fills it in


@dataclass
class HistoricalMatchEvidence(Evidence):
    """From HistoricalMatchPipeline."""

    has_history: bool
    evidence_ids: list[str]           # message_ids used as evidence
    positive_signals: int             # opened + replied
    negative_signals: int             # dismissed + muted + reported
    pattern_label: str                # 'trusted_sender' | 'ignored_sender' | 'muted_sender' | 'reported_sender' | 'mixed' | 'no_history'


@dataclass
class LLMClassifierEvidence(Evidence):
    """From LLMClassifierInference. Only populated when LLM is enabled."""

    message_type: str                 # from allowed values
    semantic_urgency: float           # 0–1; model's assessment
    model_reasoning: str              # internal reasoning; not written to output
    used_llm: bool                    # False when --no-llm or fallback


# ---------------------------------------------------------------------------
# Bundle and Decision — the final data structures
# ---------------------------------------------------------------------------


@dataclass
class EvidenceBundle:
    """All evidence gathered for a single IncomingMessage."""

    message: IncomingMessage
    scam_risk: ScamRiskEvidence
    sender_trust: SenderTrustEvidence
    user_behavior: UserBehaviorEvidence
    content_signals: ContentSignalsEvidence
    media_signals: MediaSignalsEvidence
    historical_match: HistoricalMatchEvidence
    llm_classifier: LLMClassifierEvidence

    # Optional — only present for group/business messages
    group_context: Optional[GroupContextEvidence] = None
    business_context: Optional[BusinessContextEvidence] = None


@dataclass
class RoutingDecision:
    """Final output for one message — maps 1:1 to a row in output.csv."""

    message_id: str
    action: str                       # notify | digest | mute
    message_type: str                 # from allowed values
    reason: str                       # short human-readable explanation
    confidence: float                 # 0–1
    evidence_message_ids: str         # semicolon-separated or 'none'
