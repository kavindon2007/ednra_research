from pipelines.base import EvidencePipeline
from context.models import IncomingMessage, ContentSignalsEvidence
from context.loader import DataContext

class ContentSignalsPipeline(EvidencePipeline[ContentSignalsEvidence]):
    def run(self, message: IncomingMessage, context: DataContext = None) -> ContentSignalsEvidence:
        text = message.message_text.lower()
        patterns = []
        
        has_payment = False
        if "pay" in text or "invoice" in text or "payment" in text or "due" in text or "transfer" in text:
            has_payment = True
            patterns.append("payment_keyword")
            
        has_event = False
        if "event" in text or "meeting" in text or "appointment" in text or "schedule" in text:
            has_event = True
            patterns.append("event_keyword")
            
        if "urgent" in text or "asap" in text or "emergency" in text or "immediate" in text:
            patterns.append("urgency_keyword")
            
        is_forward_chain = message.forwarded_count > 2
        if is_forward_chain:
            patterns.append("forward_chain")
            
        is_greeting_forward = False
        if is_forward_chain and ("good morning" in text or "happy" in text or "blessed" in text):
            is_greeting_forward = True
            patterns.append("greeting_forward")
            
        has_direct_mention = False
        if message.user_id and message.user_id in message.message_text:
            has_direct_mention = True
            patterns.append("direct_mention")
            
        urgency_score = 0.0
        if "urgency_keyword" in patterns:
            urgency_score += 0.5
        if has_payment:
            urgency_score += 0.3
        if has_direct_mention:
            urgency_score += 0.2
            
        return ContentSignalsEvidence(
            urgency_score=min(urgency_score, 1.0),
            detected_patterns=patterns,
            is_forward_chain=is_forward_chain,
            is_greeting_forward=is_greeting_forward,
            has_payment_keyword=has_payment,
            has_event_keyword=has_event,
            has_direct_mention=has_direct_mention
        )
