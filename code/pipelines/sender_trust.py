from pipelines.base import EvidencePipeline
from context.models import IncomingMessage, SenderTrustEvidence
from context.loader import DataContext

class SenderTrustPipeline(EvidencePipeline[SenderTrustEvidence]):
    def run(self, message: IncomingMessage, context: DataContext) -> SenderTrustEvidence:
        is_first_contact = True
        sender_reply_rate = 0.0
        trust_level = "unknown"
        is_admin = False
        
        if message.group_id:
            membership = context.group_members.get((message.group_id, message.user_id))
            if membership and membership.role == "admin":
                is_admin = True
                
        is_direct_mention = False
        if message.user_id in message.message_text:
            is_direct_mention = True
            
        history = [msg for msg in context.historical_messages.values() 
                   if msg.user_id == message.user_id and (
                       (message.sender_user_id and msg.sender_user_id == message.sender_user_id) or
                       (message.business_id and msg.business_id == message.business_id)
                   )]
                   
        if history:
            is_first_contact = False
            total = len(history)
            replies = 0
            for msg in history:
                events = context.message_events.get(msg.message_id)
                if events and events.message_replied:
                    replies += 1
            sender_reply_rate = replies / total if total > 0 else 0.0
            
            if sender_reply_rate > 0.5:
                trust_level = "high"
            elif sender_reply_rate > 0.1:
                trust_level = "medium"
            else:
                trust_level = "low"
                
        if message.business_id:
            biz = context.businesses.get(message.business_id)
            if biz and biz.verified:
                trust_level = "high"
                
        return SenderTrustEvidence(
            trust_level=trust_level,
            is_admin=is_admin,
            is_direct_mention=is_direct_mention,
            is_first_contact=is_first_contact,
            sender_reply_rate=sender_reply_rate
        )
