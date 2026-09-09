from context.models import EvidenceBundle
from inference.states import (
    InferredMessageState,
    ContentRiskState, SenderCredibilityState, MessageIntentState,
    UserReceptivityState, ContextualUrgencyState
)
from policy.dsi import DSI
from policy.models import PolicyDecision
import logging

logger = logging.getLogger("orchestrator")

class PolicyEngine:
    @classmethod
    def decide(cls, state: InferredMessageState, bundle: EvidenceBundle) -> tuple[str, PolicyDecision]:
        action = "digest"
        policy_id = "DEFAULT"
        reason = "Safe fallback"
        priority = 0
        
        # Absolute Policies
        if state.content_risk == ContentRiskState.INJECTION:
            if state.sender_credibility not in (SenderCredibilityState.KNOWN_TRUSTED, SenderCredibilityState.VERIFIED_TRUSTED):
                return "mute", PolicyDecision(policy_name="AP-1", explanation="Prompt injection from untrusted sender", priority=100)
                
        if state.content_risk == ContentRiskState.SCAM and state.sender_credibility == SenderCredibilityState.IMPERSONATOR:
            return "mute", PolicyDecision(policy_name="AP-2", explanation="Scam content from impersonator", priority=99)
            
        # Contextual Policies (Negative tier)
        if state.content_risk in (ContentRiskState.SCAM, ContentRiskState.PHISHING) and \
           state.sender_credibility in (SenderCredibilityState.IMPERSONATOR, SenderCredibilityState.KNOWN_NEGATIVE, SenderCredibilityState.UNKNOWN):
            action, policy_id, reason, priority = "mute", "CP-1", "Fraudulent content from unknown/negative sender", 90
            
        elif state.user_receptivity == UserReceptivityState.ACTIVELY_AVOIDING and state.contextual_urgency != ContextualUrgencyState.CRITICAL:
            action, policy_id, reason, priority = "mute", "CP-2", "User actively avoiding source", 89
            
        elif state.user_receptivity == UserReceptivityState.OPTED_OUT and state.message_intent == MessageIntentState.PROMOTIONAL:
            action, policy_id, reason, priority = "mute", "CP-3", "Opted out of promotional content", 88
            
        elif state.message_intent == MessageIntentState.FORWARD_CHAIN:
            action, policy_id, reason, priority = "mute", "CP-4", "Chain forward detected", 87
            
        elif state.user_receptivity == UserReceptivityState.FATIGUED and \
           state.contextual_urgency in (ContextualUrgencyState.NONE, ContextualUrgencyState.LOW) and \
           state.message_intent in (MessageIntentState.PROMOTIONAL, MessageIntentState.FORWARD_CHAIN, MessageIntentState.SOCIAL):
            action, policy_id, reason, priority = "mute", "CP-5", "Fatigued user receiving low value content", 86
            
        elif state.content_risk in (ContentRiskState.SCAM, ContentRiskState.PHISHING) and \
           state.sender_credibility == SenderCredibilityState.KNOWN_NEUTRAL and \
           bundle.historical_match.negative_signals > bundle.historical_match.positive_signals:
            action, policy_id, reason, priority = "mute", "CP-6", "Scam content with net-negative history", 85

        # Contextual Policies (Positive tier)
        elif state.contextual_urgency == ContextualUrgencyState.CRITICAL and state.sender_credibility >= SenderCredibilityState.KNOWN_NEUTRAL:
            action, policy_id, reason, priority = "notify", "CP-7", "Critical urgency from credible sender", 70
            
        elif state.contextual_urgency == ContextualUrgencyState.HIGH and \
           state.user_receptivity == UserReceptivityState.RECEPTIVE and \
           state.sender_credibility >= SenderCredibilityState.KNOWN_NEUTRAL:
            action, policy_id, reason, priority = "notify", "CP-8", "High urgency and user is receptive", 69
            
        elif state.sender_credibility in (SenderCredibilityState.KNOWN_TRUSTED, SenderCredibilityState.VERIFIED_TRUSTED) and \
           bundle.historical_match.pattern_label == 'trusted_sender' and \
           state.contextual_urgency in (ContextualUrgencyState.HIGH, ContextualUrgencyState.CRITICAL, ContextualUrgencyState.MEDIUM):
            action, policy_id, reason, priority = "notify", "CP-9", "Trusted sender with meaningful urgency", 68
            
        elif state.user_receptivity == UserReceptivityState.ACTIVELY_AVOIDING and state.contextual_urgency == ContextualUrgencyState.CRITICAL:
            action, policy_id, reason, priority = "notify", "CP-10", "Critical override for avoided source", 67
            
        # Contextual Policies (Digest tier)
        elif state.user_receptivity == UserReceptivityState.DND:
            action, policy_id, reason, priority = "digest", "CP-11", "User in Do Not Disturb window", 50
            
        elif state.message_intent == MessageIntentState.OPERATIONAL and state.sender_credibility >= SenderCredibilityState.KNOWN_NEUTRAL:
            action, policy_id, reason, priority = "digest", "CP-12", "Operational content from credible sender", 49
            
        elif state.message_intent in (MessageIntentState.INFORMATIONAL, MessageIntentState.SOCIAL) and state.sender_credibility >= SenderCredibilityState.KNOWN_NEUTRAL:
            action, policy_id, reason, priority = "digest", "CP-13", "Useful content from credible sender", 48
            
        elif state.message_intent == MessageIntentState.PROMOTIONAL and \
           state.sender_credibility >= SenderCredibilityState.KNOWN_NEUTRAL and \
           state.user_receptivity == UserReceptivityState.RECEPTIVE:
            action, policy_id, reason, priority = "digest", "CP-14", "Promotional content to receptive user", 47

        return action, PolicyDecision(policy_name=policy_id, explanation=f"[{policy_id}] {reason}", priority=priority)

