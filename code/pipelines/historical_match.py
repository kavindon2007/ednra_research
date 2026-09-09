from pipelines.base import EvidencePipeline
from context.models import IncomingMessage, HistoricalMatchEvidence
from context.loader import DataContext

class HistoricalMatchPipeline(EvidencePipeline[HistoricalMatchEvidence]):
    def run(self, message: IncomingMessage, context: DataContext) -> HistoricalMatchEvidence:
        history = [msg for msg in context.historical_messages.values() 
                   if msg.user_id == message.user_id and (
                       (message.sender_user_id and msg.sender_user_id == message.sender_user_id) or
                       (message.business_id and msg.business_id == message.business_id)
                   )]
                   
        has_history = len(history) > 0
        evidence_ids = [msg.message_id for msg in history]
        
        pos = 0
        neg = 0
        for msg in history:
            events = context.message_events.get(msg.message_id)
            if events:
                if events.message_replied or events.message_opened:
                    pos += 1
                if events.notification_dismissed or events.muted_after_message or events.message_reported:
                    neg += 1
                    
        pattern_label = "no_history"
        if has_history:
            if pos == 0 and neg > 0:
                pattern_label = "ignored_sender"
                for msg in history:
                    e = context.message_events.get(msg.message_id)
                    if e and e.message_reported:
                        pattern_label = "reported_sender"
                        break
                    elif e and e.muted_after_message:
                        if pattern_label != "reported_sender":
                            pattern_label = "muted_sender"
            elif pos > neg:
                pattern_label = "trusted_sender"
            else:
                pattern_label = "mixed"
                
        return HistoricalMatchEvidence(
            has_history=has_history,
            evidence_ids=evidence_ids,
            positive_signals=pos,
            negative_signals=neg,
            pattern_label=pattern_label
        )
