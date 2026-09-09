"""
output/trace_builder.py

TraceBuilder: constructs a ReasoningTrace from the outputs of the inference
engine and policy engine.

FAITHFULNESS GUARANTEE
──────────────────────
Every claim in the produced ReasoningTrace is traceable to an actual
computation result:

  evidence_chain       ← reads EvidenceBundle fields directly; no invention
  inference_chain      ← reads InferredMessageState trace strings; set by
                         inference engine guard logic, not by this module
  policy_fired         ← policy_id is passed by the policy engine (the rule
                         that actually fired), not inferred from the action
  dissenting_evidence  ← only added when counter-evidence actually exists in
                         the EvidenceBundle; never fabricated
  evidence_message_ids ← exact copy of HistoricalMatchEvidence.evidence_ids
  reason               ← built from a template keyed on policy_id, augmented
                         with the actual inference trace; never LLM-generated

This design ensures the explanation cannot diverge from the decision process:
changing the inference rules or policy rules automatically changes the trace.
"""

from __future__ import annotations

from typing import Optional

from context.models import (
    EvidenceBundle,
    ScamRiskEvidence,
    SenderTrustEvidence,
    UserBehaviorEvidence,
    GroupContextEvidence,
    BusinessContextEvidence,
    ContentSignalsEvidence,
    MediaSignalsEvidence,
    HistoricalMatchEvidence,
    LLMClassifierEvidence,
)
from inference.states import (
    InferredMessageState,
    ContentRiskState,
    SenderCredibilityState,
    MessageIntentState,
    UserReceptivityState,
    ContextualUrgencyState,
)
from output.trace import (
    EvidenceStep,
    InferenceStep,
    DissentStep,
    ReasoningTrace,
)
from policy.models import PolicyDecision





# ---------------------------------------------------------------------------
# Evidence inputs per state — used for InferenceStep.evidence_inputs
# ---------------------------------------------------------------------------

_STATE_EVIDENCE_INPUTS: dict[str, tuple[str, ...]] = {
    "ContentRiskState": (
        "ScamRiskEvidence.risk_level",
        "ScamRiskEvidence.flags",
        "ContentSignalsEvidence.is_forward_chain",
        "ContentSignalsEvidence.is_greeting_forward",
    ),
    "SenderCredibilityState": (
        "BusinessContextEvidence.is_verified",
        "ScamRiskEvidence.flags",
        "SenderTrustEvidence.trust_level",
        "SenderTrustEvidence.is_first_contact",
        "HistoricalMatchEvidence.pattern_label",
        "HistoricalMatchEvidence.negative_signals",
        "HistoricalMatchEvidence.positive_signals",
    ),
    "MessageIntentState": (
        "ContentSignalsEvidence.is_forward_chain",
        "ContentSignalsEvidence.is_greeting_forward",
        "ContentSignalsEvidence.urgency_score",
        "ContentSignalsEvidence.has_direct_mention",
        "ContentSignalsEvidence.has_payment_keyword",
        "ContentSignalsEvidence.has_event_keyword",
        "LLMClassifierEvidence.message_type",
        "LLMClassifierEvidence.semantic_urgency",
    ),
    "UserReceptivityState": (
        "GroupContextEvidence.is_muted_by_user",
        "HistoricalMatchEvidence.pattern_label",
        "BusinessContextEvidence.opted_out",
        "UserBehaviorEvidence.in_dnd",
        "UserBehaviorEvidence.dismiss_rate",
        "UserBehaviorEvidence.fatigue_score",
    ),
    "ContextualUrgencyState": (
        "ContentSignalsEvidence.urgency_score",
        "SenderTrustEvidence.is_direct_mention",
        "MediaSignalsEvidence.inferred_urgency",
        "LLMClassifierEvidence.semantic_urgency",
    ),
}


# ---------------------------------------------------------------------------
# TraceBuilder
# ---------------------------------------------------------------------------

class TraceBuilder:
    """
    Constructs a ReasoningTrace from the outputs of the inference and policy
    engines.

    This class is stateless. Call build() once per message.
    """

    def build(
        self,
        bundle: EvidenceBundle,
        state: InferredMessageState,
        decision: PolicyDecision,
        action: str,
        message_type: str,
        confidence: float,
    ) -> ReasoningTrace:
        """
        Build and return a ReasoningTrace.

        Parameters
        ----------
        bundle       : complete EvidenceBundle for this message
        state        : InferredMessageState produced by InferenceEngine
        decision     : PolicyDecision returned by PolicyEngine
        action       : "notify" | "digest" | "mute"
        message_type : resolved message type string
        confidence   : DSI score (0–1)
        """
        evidence_chain = self._build_evidence_chain(bundle)
        inference_chain = self._build_inference_chain(state)
        dissenting = self._build_dissenting_evidence(bundle, state, action)
        evidence_ids = tuple(bundle.historical_match.evidence_ids)

        return ReasoningTrace(
            message_id=bundle.message.message_id,
            evidence_chain=tuple(evidence_chain),
            inference_chain=tuple(inference_chain),
            policy_decision=decision,
            dissenting_evidence=tuple(dissenting),
            evidence_message_ids=evidence_ids,
            action=action,
            message_type=message_type,
            confidence=confidence,
            reason=decision.explanation,
        )

    # ------------------------------------------------------------------
    # evidence_chain construction
    # ------------------------------------------------------------------

    def _build_evidence_chain(self, bundle: EvidenceBundle) -> list[EvidenceStep]:
        steps: list[EvidenceStep] = []

        steps.extend(self._scam_risk_steps(bundle.scam_risk))
        steps.extend(self._sender_trust_steps(bundle.sender_trust))
        steps.extend(self._user_behavior_steps(bundle.user_behavior))
        steps.extend(self._content_signals_steps(bundle.content_signals))
        steps.extend(self._historical_steps(bundle.historical_match))
        steps.extend(self._llm_steps(bundle.llm_classifier))
        steps.extend(self._media_steps(bundle.media_signals))

        if bundle.group_context is not None:
            steps.extend(self._group_context_steps(bundle.group_context))
        if bundle.business_context is not None:
            steps.extend(self._business_context_steps(bundle.business_context))

        return steps

    def _scam_risk_steps(self, ev: ScamRiskEvidence) -> list[EvidenceStep]:
        steps = []
        risk = ev.risk_level

        if risk in ("high", "medium"):
            supports = "mute"
            interp = (
                f"High-confidence fraud signals: {', '.join(ev.flags)}"
                if risk == "high"
                else f"Phishing signals present: {', '.join(ev.flags)}"
            )
        elif risk == "low":
            supports = "neutral"
            interp = f"Mild risk signals: {', '.join(ev.flags) or 'unclassified'}"
        else:
            supports = "neutral"
            interp = "No scam risk indicators"

        steps.append(EvidenceStep(
            pipeline="ScamRiskPipeline",
            field="risk_level",
            value=risk,
            interpretation=interp,
            supports_action=supports,
        ))

        if ev.flags:
            steps.append(EvidenceStep(
                pipeline="ScamRiskPipeline",
                field="flags",
                value=", ".join(ev.flags),
                interpretation="Specific risk flags raised by scam detector",
                supports_action=supports,
            ))

        return steps

    def _sender_trust_steps(self, ev: SenderTrustEvidence) -> list[EvidenceStep]:
        steps = []

        trust_to_action = {
            "high": "notify",
            "medium": "digest",
            "low": "mute",
            "unknown": "neutral",
        }
        supports = trust_to_action.get(ev.trust_level, "neutral")
        interp_map = {
            "high": "User consistently replies to this sender",
            "medium": "User has moderate engagement with this sender",
            "low": "User has low engagement or negative history with this sender",
            "unknown": "No established relationship with this sender",
        }

        steps.append(EvidenceStep(
            pipeline="SenderTrustPipeline",
            field="trust_level",
            value=ev.trust_level,
            interpretation=interp_map.get(ev.trust_level, ""),
            supports_action=supports,
        ))

        if ev.is_first_contact:
            steps.append(EvidenceStep(
                pipeline="SenderTrustPipeline",
                field="is_first_contact",
                value="True",
                interpretation="No prior message history with this sender",
                supports_action="neutral",
            ))

        if ev.is_direct_mention:
            steps.append(EvidenceStep(
                pipeline="SenderTrustPipeline",
                field="is_direct_mention",
                value="True",
                interpretation="Sender directly addressed this user by ID",
                supports_action="notify",
            ))

        if ev.sender_reply_rate > 0:
            steps.append(EvidenceStep(
                pipeline="SenderTrustPipeline",
                field="sender_reply_rate",
                value=f"{ev.sender_reply_rate:.2f}",
                interpretation=(
                    f"User replied to {ev.sender_reply_rate:.0%} of past messages "
                    "from this sender"
                ),
                supports_action="notify" if ev.sender_reply_rate > 0.5 else "neutral",
            ))

        return steps

    def _user_behavior_steps(self, ev: UserBehaviorEvidence) -> list[EvidenceStep]:
        steps = []

        steps.append(EvidenceStep(
            pipeline="UserBehaviorPipeline",
            field="in_dnd",
            value=str(ev.in_dnd),
            interpretation=(
                "Message arrived inside the user's do-not-disturb window"
                if ev.in_dnd
                else "Message arrived outside the user's DND window"
            ),
            supports_action="mute" if ev.in_dnd else "neutral",
        ))

        if ev.dismiss_rate > 0:
            supports = "mute" if ev.dismiss_rate > 0.7 else "neutral"
            steps.append(EvidenceStep(
                pipeline="UserBehaviorPipeline",
                field="dismiss_rate",
                value=f"{ev.dismiss_rate:.2f}",
                interpretation=(
                    f"User dismisses {ev.dismiss_rate:.0%} of notifications (high fatigue)"
                    if ev.dismiss_rate > 0.7
                    else f"User dismisses {ev.dismiss_rate:.0%} of notifications (normal)"
                ),
                supports_action=supports,
            ))

        if ev.fatigue_score > 0:
            steps.append(EvidenceStep(
                pipeline="UserBehaviorPipeline",
                field="fatigue_score",
                value=f"{ev.fatigue_score:.2f}",
                interpretation=(
                    "High recent notification fatigue from daily summary"
                    if ev.fatigue_score > 0.5
                    else "Moderate recent notification fatigue"
                ),
                supports_action="mute" if ev.fatigue_score > 0.5 else "neutral",
            ))

        return steps

    def _content_signals_steps(self, ev: ContentSignalsEvidence) -> list[EvidenceStep]:
        steps = []

        if ev.urgency_score > 0:
            supports = "notify" if ev.urgency_score > 0.6 else "neutral"
            steps.append(EvidenceStep(
                pipeline="ContentSignalsPipeline",
                field="urgency_score",
                value=f"{ev.urgency_score:.2f}",
                interpretation=(
                    f"High urgency score ({ev.urgency_score:.2f}) — time-sensitive language detected"
                    if ev.urgency_score > 0.6
                    else f"Low urgency score ({ev.urgency_score:.2f})"
                ),
                supports_action=supports,
            ))

        if ev.detected_patterns:
            steps.append(EvidenceStep(
                pipeline="ContentSignalsPipeline",
                field="detected_patterns",
                value=", ".join(ev.detected_patterns),
                interpretation=f"Text heuristics matched: {', '.join(ev.detected_patterns)}",
                supports_action=(
                    "mute"
                    if any(p in ev.detected_patterns for p in (
                        "otp_request", "credential_request", "injection_attack",
                        "account_block_threat", "urgent_keyword",
                    ))
                    else "neutral"
                ),
            ))

        if ev.is_forward_chain:
            steps.append(EvidenceStep(
                pipeline="ContentSignalsPipeline",
                field="is_forward_chain",
                value="True",
                interpretation="Message is part of a viral forward chain",
                supports_action="mute",
            ))

        return steps

    def _historical_steps(self, ev: HistoricalMatchEvidence) -> list[EvidenceStep]:
        steps = []

        if not ev.has_history:
            steps.append(EvidenceStep(
                pipeline="HistoricalMatchPipeline",
                field="pattern_label",
                value="no_history",
                interpretation="No prior message history with this sender",
                supports_action="neutral",
            ))
            return steps

        label_to_action = {
            "trusted_sender": "notify",
            "mixed": "neutral",
            "ignored_sender": "mute",
            "muted_sender": "mute",
            "reported_sender": "mute",
            "no_history": "neutral",
        }
        steps.append(EvidenceStep(
            pipeline="HistoricalMatchPipeline",
            field="pattern_label",
            value=ev.pattern_label,
            interpretation=(
                f"Historical pattern: {ev.pattern_label.replace('_', ' ')} "
                f"({ev.positive_signals} positive, {ev.negative_signals} negative signals)"
            ),
            supports_action=label_to_action.get(ev.pattern_label, "neutral"),
        ))

        return steps

    def _llm_steps(self, ev: LLMClassifierEvidence) -> list[EvidenceStep]:
        steps = []

        if ev.used_llm:
            steps.append(EvidenceStep(
                pipeline="LLMClassifierInference",
                field="message_type",
                value=ev.message_type,
                interpretation=f"LLM classified as: {ev.message_type}",
                supports_action=(
                    "notify" if ev.semantic_urgency > 0.7
                    else "mute" if ev.message_type in ("scam", "spam")
                    else "digest"
                ),
            ))

            if ev.semantic_urgency > 0:
                steps.append(EvidenceStep(
                    pipeline="LLMClassifierInference",
                    field="semantic_urgency",
                    value=f"{ev.semantic_urgency:.2f}",
                    interpretation=f"LLM semantic urgency: {ev.semantic_urgency:.2f}",
                    supports_action=(
                        "notify" if ev.semantic_urgency > 0.7 else "neutral"
                    ),
                ))
        else:
            steps.append(EvidenceStep(
                pipeline="LLMClassifierInference",
                field="used_llm",
                value="False",
                interpretation="LLM not used; message_type is a deterministic fallback",
                supports_action="neutral",
            ))

        return steps

    def _media_steps(self, ev: MediaSignalsEvidence) -> list[EvidenceStep]:
        steps = []

        if ev.has_media:
            steps.append(EvidenceStep(
                pipeline="MediaSignalsPipeline",
                field="media_type",
                value=ev.media_type,
                interpretation=f"Message contains {ev.media_type} media",
                supports_action="neutral",
            ))

            if ev.media_description:
                steps.append(EvidenceStep(
                    pipeline="MediaSignalsPipeline",
                    field="media_description",
                    value=ev.media_description[:80] + ("…" if len(ev.media_description) > 80 else ""),
                    interpretation="LLM-generated description of media content",
                    supports_action=(
                        "notify" if ev.inferred_urgency > 0.6 else "neutral"
                    ),
                ))

        return steps

    def _group_context_steps(self, ev: GroupContextEvidence) -> list[EvidenceStep]:
        steps = []

        steps.append(EvidenceStep(
            pipeline="GroupContextPipeline",
            field="group_type",
            value=ev.group_type,
            interpretation=f"Message from a {ev.group_type.replace('_', ' ')} group",
            supports_action=(
                "notify" if ev.group_type in ("coworker", "family")
                else "digest"
            ),
        ))

        if ev.is_muted_by_user:
            steps.append(EvidenceStep(
                pipeline="GroupContextPipeline",
                field="is_muted_by_user",
                value="True",
                interpretation="User has muted this group",
                supports_action="mute",
            ))

        if ev.user_engagement_ratio > 0:
            steps.append(EvidenceStep(
                pipeline="GroupContextPipeline",
                field="user_engagement_ratio",
                value=f"{ev.user_engagement_ratio:.2f}",
                interpretation=(
                    f"User replies to {ev.user_engagement_ratio:.0%} of messages in this group"
                ),
                supports_action=(
                    "notify" if ev.user_engagement_ratio > 0.4 else "neutral"
                ),
            ))

        return steps

    def _business_context_steps(self, ev: BusinessContextEvidence) -> list[EvidenceStep]:
        steps = []

        steps.append(EvidenceStep(
            pipeline="BusinessContextPipeline",
            field="is_verified",
            value=str(ev.is_verified),
            interpretation=(
                "Business is platform-verified" if ev.is_verified
                else "Business is NOT verified on this platform"
            ),
            supports_action="notify" if ev.is_verified else "neutral",
        ))

        if ev.opted_out:
            steps.append(EvidenceStep(
                pipeline="BusinessContextPipeline",
                field="opted_out",
                value="True",
                interpretation="User explicitly opted out of promotions from this business",
                supports_action="mute",
            ))

        if ev.has_active_relation:
            steps.append(EvidenceStep(
                pipeline="BusinessContextPipeline",
                field="has_active_relation",
                value="True",
                interpretation=(
                    f"User has an active relationship with this business: "
                    f"{ev.relation_type.replace('_', ' ')}"
                ),
                supports_action="digest",
            ))

        return steps

    # ------------------------------------------------------------------
    # inference_chain construction
    # ------------------------------------------------------------------

    def _build_inference_chain(self, state: InferredMessageState) -> list[InferenceStep]:
        """
        Each InferenceStep maps one state in InferredMessageState to the
        guard that produced it and the evidence sources it consumed.
        """
        steps = [
            InferenceStep(
                state_name="ContentRiskState",
                state_value=state.content_risk.value.upper(),
                guard=state.content_risk_trace,
                evidence_inputs=_STATE_EVIDENCE_INPUTS["ContentRiskState"],
            ),
            InferenceStep(
                state_name="SenderCredibilityState",
                state_value=state.sender_credibility.value.upper(),
                guard=state.sender_credibility_trace,
                evidence_inputs=_STATE_EVIDENCE_INPUTS["SenderCredibilityState"],
            ),
            InferenceStep(
                state_name="MessageIntentState",
                state_value=state.message_intent.value.upper(),
                guard=state.message_intent_trace,
                evidence_inputs=_STATE_EVIDENCE_INPUTS["MessageIntentState"],
            ),
            InferenceStep(
                state_name="UserReceptivityState",
                state_value=state.user_receptivity.value.upper(),
                guard=state.user_receptivity_trace,
                evidence_inputs=_STATE_EVIDENCE_INPUTS["UserReceptivityState"],
            ),
            InferenceStep(
                state_name="ContextualUrgencyState",
                state_value=state.contextual_urgency.value.upper(),
                guard=state.contextual_urgency_trace,
                evidence_inputs=_STATE_EVIDENCE_INPUTS["ContextualUrgencyState"],
            ),
        ]
        return steps



    # ------------------------------------------------------------------
    # dissenting_evidence construction
    # ------------------------------------------------------------------

    def _build_dissenting_evidence(
        self,
        bundle: EvidenceBundle,
        state: InferredMessageState,
        action: str,
    ) -> list[DissentStep]:
        """
        Identify evidence that actually exists in the bundle but points
        in the OPPOSITE direction from the taken action.

        Each dissent check is a deterministic predicate — no LLM involved.
        """
        dissents: list[DissentStep] = []

        if action == "mute":
            dissents.extend(self._mute_dissents(bundle, state))
        elif action == "notify":
            dissents.extend(self._notify_dissents(bundle, state))
        elif action == "digest":
            dissents.extend(self._digest_dissents(bundle, state))

        return dissents

    def _mute_dissents(
        self, bundle: EvidenceBundle, state: InferredMessageState
    ) -> list[DissentStep]:
        dissents = []

        # Trusted sender history argues against muting
        if bundle.historical_match.pattern_label == "trusted_sender":
            dissents.append(DissentStep(
                source="HistoricalMatchPipeline",
                field="pattern_label",
                value="trusted_sender",
                argues_for="notify",
                overrule_reason=(
                    "Content risk or sender impersonation outweighs historical trust"
                ),
            ))

        # High urgency argues against muting
        if state.contextual_urgency in (
            ContextualUrgencyState.CRITICAL, ContextualUrgencyState.HIGH
        ):
            dissents.append(DissentStep(
                source="ContextualUrgencyState (inferred)",
                field="contextual_urgency",
                value=state.contextual_urgency.value,
                argues_for="notify",
                overrule_reason=(
                    "Sender credibility is insufficient for urgency to override mute"
                ),
            ))

        # Direct mention argues against muting
        if bundle.sender_trust.is_direct_mention:
            dissents.append(DissentStep(
                source="SenderTrustPipeline",
                field="is_direct_mention",
                value="True",
                argues_for="notify",
                overrule_reason=(
                    "Direct mention insufficient when content risk or avoidance is high"
                ),
            ))

        return dissents

    def _notify_dissents(
        self, bundle: EvidenceBundle, state: InferredMessageState
    ) -> list[DissentStep]:
        dissents = []

        # DND argues against notifying
        if bundle.user_behavior.in_dnd:
            dissents.append(DissentStep(
                source="UserBehaviorPipeline",
                field="in_dnd",
                value="True",
                argues_for="digest",
                overrule_reason=(
                    "Critical urgency overrides DND restriction"
                ),
            ))

        # Suspicious content argues against notifying
        if state.content_risk in (ContentRiskState.SUSPICIOUS, ContentRiskState.SPAM):
            dissents.append(DissentStep(
                source="ContentRiskState (inferred)",
                field="content_risk",
                value=state.content_risk.value,
                argues_for="mute",
                overrule_reason=(
                    "Trusted sender history outweighs mild content risk"
                ),
            ))

        # High dismiss rate argues against notifying
        if bundle.user_behavior.dismiss_rate > 0.6:
            dissents.append(DissentStep(
                source="UserBehaviorPipeline",
                field="dismiss_rate",
                value=f"{bundle.user_behavior.dismiss_rate:.2f}",
                argues_for="digest",
                overrule_reason=(
                    "Urgency or trusted sender outweighs fatigue signal"
                ),
            ))

        return dissents

    def _digest_dissents(
        self, bundle: EvidenceBundle, state: InferredMessageState
    ) -> list[DissentStep]:
        dissents = []

        # High urgency argues for notify instead
        if state.contextual_urgency in (
            ContextualUrgencyState.CRITICAL, ContextualUrgencyState.HIGH
        ):
            dissents.append(DissentStep(
                source="ContextualUrgencyState (inferred)",
                field="contextual_urgency",
                value=state.contextual_urgency.value,
                argues_for="notify",
                overrule_reason=(
                    "Sender credibility or user receptivity insufficient to elevate to notify"
                ),
            ))

        # Scam/phishing content argues for mute instead
        if state.content_risk >= ContentRiskState.PHISHING:
            dissents.append(DissentStep(
                source="ContentRiskState (inferred)",
                field="content_risk",
                value=state.content_risk.value,
                argues_for="mute",
                overrule_reason=(
                    "Trusted sender history or verified business status mitigates content risk"
                ),
            ))

        return dissents
