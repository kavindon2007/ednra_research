from context.models import EvidenceBundle
from inference.states import (
    InferredMessageState,
    ContentRiskState, SenderCredibilityState, MessageIntentState,
    UserReceptivityState, ContextualUrgencyState, GuardFired
)

class InferenceEngine:
    @classmethod
    def infer(cls, bundle: EvidenceBundle) -> InferredMessageState:
        # 1. Content Risk
        risk_state = ContentRiskState.CLEAN
        risk_trace = GuardFired("Default", "No risk detected")
        
        if bundle.scam_risk.risk_level == "high":
            risk_state = ContentRiskState.SCAM
            risk_trace = GuardFired("ScamRisk", "High confidence fraud signals")
        elif bundle.scam_risk.risk_level == "medium":
            risk_state = ContentRiskState.PHISHING
            risk_trace = GuardFired("ScamRisk", "Medium confidence phishing/fraud")
        elif "injection" in bundle.content_signals.detected_patterns: # Placeholder for injection
            risk_state = ContentRiskState.INJECTION
            risk_trace = GuardFired("ContentSignals", "Prompt injection detected")
        elif bundle.content_signals.is_forward_chain:
            risk_state = ContentRiskState.SPAM
            risk_trace = GuardFired("ContentSignals", "Viral chain forward")
        elif bundle.scam_risk.risk_level == "low":
            risk_state = ContentRiskState.SUSPICIOUS
            risk_trace = GuardFired("ScamRisk", "Suspicious pattern detected")

        # 2. Sender Credibility
        cred_state = SenderCredibilityState.UNKNOWN
        cred_trace = GuardFired("Default", "No history")
        
        if bundle.historical_match.pattern_label in ("muted_sender", "reported_sender") or bundle.sender_trust.trust_level == "low":
            cred_state = SenderCredibilityState.KNOWN_NEGATIVE
            cred_trace = GuardFired("SenderTrust", "Sender has been muted, reported, or has very low trust")
        elif bundle.sender_trust.trust_level == "high" or bundle.business_context and bundle.business_context.is_verified:
            if bundle.business_context and bundle.business_context.is_verified:
                cred_state = SenderCredibilityState.VERIFIED_TRUSTED
                cred_trace = GuardFired("BusinessContext", "Verified business")
            else:
                cred_state = SenderCredibilityState.KNOWN_TRUSTED
                cred_trace = GuardFired("SenderTrust", "High trust or frequent positive interaction")
        elif bundle.historical_match.pattern_label in ("mixed", "ignored_sender"):
            cred_state = SenderCredibilityState.KNOWN_NEUTRAL
            cred_trace = GuardFired("HistoricalMatch", "Known sender but mixed/neutral engagement")

        if bundle.scam_risk.risk_level == "high" and bundle.business_context and not bundle.business_context.is_verified:
            cred_state = SenderCredibilityState.IMPERSONATOR
            cred_trace = GuardFired("ScamRisk", "Unverified business with domain mismatch")

        # 3. Message Intent
        intent_state = MessageIntentState.AMBIGUOUS
        intent_trace = GuardFired("Default", "No clear intent")
        
        if bundle.llm_classifier.used_llm:
            t = bundle.llm_classifier.message_type
            if t == "transactional" or t == "otp":
                intent_state = MessageIntentState.OPERATIONAL
                intent_trace = GuardFired("LLM", f"Classified as {t}")
            elif t == "promotion":
                intent_state = MessageIntentState.PROMOTIONAL
                intent_trace = GuardFired("LLM", "Classified as promotional")
            elif t == "reminder":
                intent_state = MessageIntentState.URGENT
                intent_trace = GuardFired("LLM", "Classified as reminder/urgent")
            elif t in ("personal", "group_chat"):
                intent_state = MessageIntentState.SOCIAL
                intent_trace = GuardFired("LLM", f"Classified as {t}")
                
        if intent_state == MessageIntentState.AMBIGUOUS:
            if bundle.content_signals.has_event_keyword or bundle.content_signals.has_payment_keyword:
                intent_state = MessageIntentState.OPERATIONAL
                intent_trace = GuardFired("ContentSignals", "Contains event or payment keywords")
            if bundle.content_signals.is_forward_chain:
                intent_state = MessageIntentState.FORWARD_CHAIN
                intent_trace = GuardFired("ContentSignals", "Viral chain forward")

        # 4. User Receptivity
        rec_state = UserReceptivityState.RECEPTIVE
        rec_trace = GuardFired("Default", "User is receptive")
        
        if bundle.business_context and bundle.business_context.opted_out:
            rec_state = UserReceptivityState.OPTED_OUT
            rec_trace = GuardFired("BusinessContext", "User opted out")
        elif (bundle.group_context and bundle.group_context.is_muted_by_user) or bundle.historical_match.pattern_label == "muted_sender":
            rec_state = UserReceptivityState.ACTIVELY_AVOIDING
            rec_trace = GuardFired("Group/History", "User explicitly muted this source")
        elif bundle.user_behavior.in_dnd:
            rec_state = UserReceptivityState.DND
            rec_trace = GuardFired("UserBehavior", "Inside DND window")
        elif bundle.user_behavior.fatigue_score > 0.8:
            rec_state = UserReceptivityState.FATIGUED
            rec_trace = GuardFired("UserBehavior", "High notification fatigue")

        # 5. Contextual Urgency
        urgency_state = ContextualUrgencyState.NONE
        urgency_trace = GuardFired("Default", "No urgency detected")
        
        u_score = bundle.content_signals.urgency_score
        if bundle.llm_classifier.used_llm:
            u_score = max(u_score, bundle.llm_classifier.semantic_urgency)
            
        if u_score > 0.8:
            urgency_state = ContextualUrgencyState.CRITICAL
            urgency_trace = GuardFired("Signals", "Very high urgency score")
        elif u_score > 0.5:
            urgency_state = ContextualUrgencyState.HIGH
            urgency_trace = GuardFired("Signals", "High urgency score")
        elif u_score > 0.2:
            urgency_state = ContextualUrgencyState.MEDIUM
            urgency_trace = GuardFired("Signals", "Medium urgency score")
        elif u_score > 0:
            urgency_state = ContextualUrgencyState.LOW
            urgency_trace = GuardFired("Signals", "Low urgency score")

        return InferredMessageState(
            content_risk=risk_state,
            sender_credibility=cred_state,
            message_intent=intent_state,
            user_receptivity=rec_state,
            contextual_urgency=urgency_state,
            content_risk_trace=risk_trace,
            sender_credibility_trace=cred_trace,
            message_intent_trace=intent_trace,
            user_receptivity_trace=rec_trace,
            contextual_urgency_trace=urgency_trace,
            injection_from_trusted=False
        )
