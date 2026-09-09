"""
tests/helpers.py

Shared factory functions for constructing test fixtures.

All factory functions produce minimal, valid objects with safe defaults.
Tests override only the fields they care about, keeping tests readable
and focussed on one behaviour at a time.
"""

from __future__ import annotations

from context.models import (
    IncomingMessage,
    ScamRiskEvidence,
    SenderTrustEvidence,
    UserBehaviorEvidence,
    GroupContextEvidence,
    BusinessContextEvidence,
    ContentSignalsEvidence,
    MediaSignalsEvidence,
    HistoricalMatchEvidence,
    LLMClassifierEvidence,
    EvidenceBundle,
)
from inference.states import (
    InferredMessageState,
    ContentRiskState,
    SenderCredibilityState,
    MessageIntentState,
    UserReceptivityState,
    ContextualUrgencyState,
    GuardFired,
)
from policy.models import PolicyDecision


# ---------------------------------------------------------------------------
# Message factory
# ---------------------------------------------------------------------------

def make_message(
    message_id: str = "msg_test",
    user_id: str = "u_001",
    conversation_type: str = "personal",
    group_id: str | None = None,
    business_id: str | None = None,
    sender_user_id: str | None = "u_002",
    created_at: str = "2026-07-01 10:00",
    message_text: str = "Hello, are you free tomorrow?",
    media_type: str = "",
    media_id: str | None = None,
    forwarded_count: int = 0,
) -> IncomingMessage:
    return IncomingMessage(
        message_id=message_id,
        user_id=user_id,
        conversation_type=conversation_type,
        group_id=group_id,
        business_id=business_id,
        sender_user_id=sender_user_id,
        created_at=created_at,
        message_text=message_text,
        media_type=media_type,
        media_id=media_id,
        forwarded_count=forwarded_count,
    )


# ---------------------------------------------------------------------------
# Evidence factories
# ---------------------------------------------------------------------------

def make_scam_risk(
    risk_level: str = "none",
    flags: list[str] | None = None,
) -> ScamRiskEvidence:
    return ScamRiskEvidence(
        risk_level=risk_level,
        flags=flags or [],
    )


def make_sender_trust(
    trust_level: str = "medium",
    is_admin: bool = False,
    is_direct_mention: bool = False,
    is_first_contact: bool = False,
    sender_reply_rate: float = 0.3,
) -> SenderTrustEvidence:
    return SenderTrustEvidence(
        trust_level=trust_level,
        is_admin=is_admin,
        is_direct_mention=is_direct_mention,
        is_first_contact=is_first_contact,
        sender_reply_rate=sender_reply_rate,
    )


def make_user_behavior(
    in_dnd: bool = False,
    dismiss_rate: float = 0.2,
    report_rate: float = 0.0,
    fatigue_score: float = 0.1,
) -> UserBehaviorEvidence:
    return UserBehaviorEvidence(
        in_dnd=in_dnd,
        dismiss_rate=dismiss_rate,
        report_rate=report_rate,
        fatigue_score=fatigue_score,
    )


def make_group_context(
    group_type: str = "coworker",
    is_muted_by_user: bool = False,
    user_engagement_ratio: float = 0.5,
    user_dismiss_rate_in_group: float = 0.1,
    sender_is_admin: bool = False,
) -> GroupContextEvidence:
    return GroupContextEvidence(
        group_type=group_type,
        is_muted_by_user=is_muted_by_user,
        user_engagement_ratio=user_engagement_ratio,
        user_dismiss_rate_in_group=user_dismiss_rate_in_group,
        sender_is_admin=sender_is_admin,
    )


def make_business_context(
    is_verified: bool = True,
    has_active_relation: bool = False,
    opted_out: bool = False,
    user_open_ratio: float = 0.5,
    relation_type: str = "",
) -> BusinessContextEvidence:
    return BusinessContextEvidence(
        is_verified=is_verified,
        has_active_relation=has_active_relation,
        opted_out=opted_out,
        user_open_ratio=user_open_ratio,
        relation_type=relation_type,
    )


def make_content_signals(
    urgency_score: float = 0.0,
    detected_patterns: list[str] | None = None,
    is_forward_chain: bool = False,
    is_greeting_forward: bool = False,
    has_payment_keyword: bool = False,
    has_event_keyword: bool = False,
    has_direct_mention: bool = False,
) -> ContentSignalsEvidence:
    return ContentSignalsEvidence(
        urgency_score=urgency_score,
        detected_patterns=detected_patterns or [],
        is_forward_chain=is_forward_chain,
        is_greeting_forward=is_greeting_forward,
        has_payment_keyword=has_payment_keyword,
        has_event_keyword=has_event_keyword,
        has_direct_mention=has_direct_mention,
    )


def make_media_signals(
    has_media: bool = False,
    media_type: str = "",
    media_id: str = "",
    media_description: str = "",
    inferred_urgency: float = 0.0,
) -> MediaSignalsEvidence:
    return MediaSignalsEvidence(
        has_media=has_media,
        media_type=media_type,
        media_id=media_id,
        media_description=media_description,
        inferred_urgency=inferred_urgency,
    )


def make_historical_match(
    has_history: bool = True,
    evidence_ids: list[str] | None = None,
    positive_signals: int = 0,
    negative_signals: int = 0,
    pattern_label: str = "no_history",
) -> HistoricalMatchEvidence:
    return HistoricalMatchEvidence(
        has_history=has_history,
        evidence_ids=evidence_ids or [],
        positive_signals=positive_signals,
        negative_signals=negative_signals,
        pattern_label=pattern_label,
    )


def make_llm_classifier(
    message_type: str = "unknown",
    semantic_urgency: float = 0.0,
    model_reasoning: str = "",
    used_llm: bool = False,
) -> LLMClassifierEvidence:
    return LLMClassifierEvidence(
        message_type=message_type,
        semantic_urgency=semantic_urgency,
        model_reasoning=model_reasoning,
        used_llm=used_llm,
    )


def make_bundle(
    message: IncomingMessage | None = None,
    scam_risk: ScamRiskEvidence | None = None,
    sender_trust: SenderTrustEvidence | None = None,
    user_behavior: UserBehaviorEvidence | None = None,
    content_signals: ContentSignalsEvidence | None = None,
    media_signals: MediaSignalsEvidence | None = None,
    historical_match: HistoricalMatchEvidence | None = None,
    llm_classifier: LLMClassifierEvidence | None = None,
    group_context: GroupContextEvidence | None = None,
    business_context: BusinessContextEvidence | None = None,
) -> EvidenceBundle:
    return EvidenceBundle(
        message=message or make_message(),
        scam_risk=scam_risk or make_scam_risk(),
        sender_trust=sender_trust or make_sender_trust(),
        user_behavior=user_behavior or make_user_behavior(),
        content_signals=content_signals or make_content_signals(),
        media_signals=media_signals or make_media_signals(),
        historical_match=historical_match or make_historical_match(),
        llm_classifier=llm_classifier or make_llm_classifier(),
        group_context=group_context,
        business_context=business_context,
    )


def make_state(
    content_risk: ContentRiskState = ContentRiskState.CLEAN,
    sender_credibility: SenderCredibilityState = SenderCredibilityState.KNOWN_NEUTRAL,
    message_intent: MessageIntentState = MessageIntentState.SOCIAL,
    user_receptivity: UserReceptivityState = UserReceptivityState.RECEPTIVE,
    contextual_urgency: ContextualUrgencyState = ContextualUrgencyState.LOW,
    content_risk_trace: GuardFired | None = None,
    sender_credibility_trace: GuardFired | None = None,
    message_intent_trace: GuardFired | None = None,
    user_receptivity_trace: GuardFired | None = None,
    contextual_urgency_trace: GuardFired | None = None,
    injection_from_trusted: bool = False,
) -> InferredMessageState:
    return InferredMessageState(
        content_risk=content_risk,
        sender_credibility=sender_credibility,
        message_intent=message_intent,
        user_receptivity=user_receptivity,
        contextual_urgency=contextual_urgency,
        content_risk_trace=content_risk_trace or GuardFired("Default", "No risk signals detected"),
        sender_credibility_trace=sender_credibility_trace or GuardFired("Default", "Known sender, neutral engagement"),
        message_intent_trace=message_intent_trace or GuardFired("Default", "Casual social message"),
        user_receptivity_trace=user_receptivity_trace or GuardFired("Default", "No suppression signals"),
        contextual_urgency_trace=contextual_urgency_trace or GuardFired("Default", "Low urgency content"),
        injection_from_trusted=injection_from_trusted,
    )


def make_policy_decision(
    policy_name: str = "Test Policy",
    explanation: str = "Test explanation",
    evidence_used: list[str] | None = None,
    priority: int = 0,
) -> PolicyDecision:
    return PolicyDecision(
        policy_name=policy_name,
        explanation=explanation,
        evidence_used=evidence_used or [],
        priority=priority,
    )
