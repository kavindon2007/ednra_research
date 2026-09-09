from pipelines.base import EvidencePipeline
from context.models import IncomingMessage, ScamRiskEvidence
from context.loader import DataContext

class ScamRiskPipeline(EvidencePipeline[ScamRiskEvidence]):
    def run(self, message: IncomingMessage, context: DataContext) -> ScamRiskEvidence:
        flags = []
        text_lower = message.message_text.lower()
        if "otp" in text_lower or "verification code" in text_lower or "password" in text_lower or "pin" in text_lower:
            flags.append("otp_request")
        
        biz = context.businesses.get(message.business_id) if message.business_id else None
        if biz and not biz.verified:
            flags.append("unverified_business")
        if biz and biz.domain_used_by_sender and biz.official_domain and biz.domain_used_by_sender != biz.official_domain:
            flags.append("domain_mismatch")
                
        risk_level = "none"
        if "domain_mismatch" in flags:
            risk_level = "high"
        elif "otp_request" in flags and ("unverified_business" in flags or not message.business_id):
            risk_level = "medium"
        elif len(flags) > 0:
            risk_level = "low"
            
        return ScamRiskEvidence(risk_level=risk_level, flags=flags)
