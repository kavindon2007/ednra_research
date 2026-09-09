from context.models import EvidenceBundle
from inference.states import InferredMessageState

FACTORS = {
    "content_risk_injection": {"weight": 1.00, "mute": 1, "notify": -1, "digest": -1},
    "content_risk_scam": {"weight": 0.95, "mute": 1, "notify": -1, "digest": -1},
    "content_risk_phishing": {"weight": 0.80, "mute": 1, "notify": -1, "digest": -1},
    "content_risk_spam": {"weight": 0.60, "mute": 1, "notify": -1, "digest": 0},
    "content_risk_suspicious": {"weight": 0.35, "mute": 1, "notify": 0, "digest": 0},
    "content_risk_clean": {"weight": 0.20, "mute": -1, "notify": 1, "digest": 1},
    "sender_impersonator": {"weight": 1.00, "mute": 1, "notify": -1, "digest": -1},
    "sender_known_negative": {"weight": 0.85, "mute": 1, "notify": -1, "digest": -1},
    "sender_unknown": {"weight": 0.25, "mute": 0, "notify": 0, "digest": 1},
    "sender_known_neutral": {"weight": 0.25, "mute": -1, "notify": 1, "digest": 1},
    "sender_known_trusted": {"weight": 0.55, "mute": -1, "notify": 1, "digest": 0},
    "sender_verified_trusted": {"weight": 0.70, "mute": -1, "notify": 1, "digest": 0},
    "urgency_critical": {"weight": 0.95, "mute": -1, "notify": 1, "digest": -1},
    "urgency_high": {"weight": 0.75, "mute": -1, "notify": 1, "digest": -1},
    "urgency_medium": {"weight": 0.40, "mute": 0, "notify": 0, "digest": 1},
    "urgency_low": {"weight": 0.30, "mute": 0, "notify": -1, "digest": 1},
    "urgency_none": {"weight": 0.35, "mute": 1, "notify": -1, "digest": 1},
    "receptivity_avoid": {"weight": 0.90, "mute": 1, "notify": -1, "digest": -1},
    "receptivity_opted_out": {"weight": 0.75, "mute": 1, "notify": -1, "digest": 0},
    "receptivity_dnd": {"weight": 0.55, "mute": 0, "notify": -1, "digest": 1},
    "receptivity_fatigued": {"weight": 0.40, "mute": 1, "notify": -1, "digest": 1},
    "receptivity_receptive": {"weight": 0.30, "mute": -1, "notify": 1, "digest": 0},
    "intent_forward_chain": {"weight": 0.70, "mute": 1, "notify": -1, "digest": -1},
    "intent_promotional": {"weight": 0.40, "mute": 1, "notify": -1, "digest": 1},
    "intent_social": {"weight": 0.25, "mute": 0, "notify": 0, "digest": 1},
    "intent_informational": {"weight": 0.20, "mute": 0, "notify": 0, "digest": 1},
    "intent_operational": {"weight": 0.30, "mute": -1, "notify": 0, "digest": 1},
    "intent_urgent": {"weight": 0.60, "mute": -1, "notify": 1, "digest": -1},
    "hist_reported": {"weight": 0.90, "mute": 1, "notify": -1, "digest": -1},
    "hist_muted": {"weight": 0.75, "mute": 1, "notify": -1, "digest": -1},
    "hist_dismissed": {"weight": 0.45, "mute": 1, "notify": -1, "digest": 0},
    "hist_replied": {"weight": 0.70, "mute": -1, "notify": 1, "digest": 0},
    "hist_opened": {"weight": 0.40, "mute": 0, "notify": 1, "digest": 1},
    "hist_no_history": {"weight": 0.20, "mute": 0, "notify": 0, "digest": 1},
}

class DSI:
    @classmethod
    def compute(cls, state: InferredMessageState, bundle: EvidenceBundle, action: str) -> float:
        active_factors = []
        
        # Content Risk
        risk_name = state.content_risk.value
        active_factors.append(f"content_risk_{risk_name}")
        
        # Sender Credibility
        cred_name = state.sender_credibility.value
        active_factors.append(f"sender_{cred_name}")
        
        # Urgency
        urgency_name = state.contextual_urgency.value
        active_factors.append(f"urgency_{urgency_name}")
        
        # Receptivity
        rec_name = state.user_receptivity.value
        if rec_name == "actively_avoiding": rec_name = "avoid"
        active_factors.append(f"receptivity_{rec_name}")
        
        # Intent
        intent_name = state.message_intent.value
        active_factors.append(f"intent_{intent_name}")
        
        # History
        hist = bundle.historical_match
        if hist.has_history:
            if hist.pattern_label == "reported_sender": active_factors.append("hist_reported")
            elif hist.pattern_label == "muted_sender": active_factors.append("hist_muted")
            elif hist.negative_signals > hist.positive_signals: active_factors.append("hist_dismissed")
            elif hist.positive_signals > hist.negative_signals: active_factors.append("hist_replied")
            else: active_factors.append("hist_opened")
        else:
            active_factors.append("hist_no_history")
            
        epsilon = 0.15
        gamma = 0.5
        
        s_plus = 0.0
        s_minus = 0.0
        w_total = 0.0
        
        for f in active_factors:
            if f in FACTORS:
                factor = FACTORS[f]
                w = factor["weight"]
                d = factor.get(action, 0)
                
                s_plus += w * max(0, d)
                s_minus += w * max(0, -d)
                w_total += w * abs(d)
                
        w_total += epsilon
        
        dsi = (s_plus - gamma * s_minus) / w_total if w_total > 0 else 0.0
        return max(0.60, min(0.97, dsi))
