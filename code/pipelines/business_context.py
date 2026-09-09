from pipelines.base import EvidencePipeline
from context.models import IncomingMessage, BusinessContextEvidence
from context.loader import DataContext

class BusinessContextPipeline(EvidencePipeline[BusinessContextEvidence | None]):
    def run(self, message: IncomingMessage, context: DataContext) -> BusinessContextEvidence | None:
        if not message.business_id:
            return None
            
        biz = context.businesses.get(message.business_id)
        if not biz:
            return None
            
        relation = context.user_business.get((message.user_id, message.business_id))
        
        is_verified = biz.verified
        has_active = False
        opted_out = False
        user_open_ratio = 0.0
        relation_type = ""
        
        if relation:
            has_active = relation.activity_count_180d > 0
            opted_out = not relation.allows_promotions or relation.promotions_opted_out_at is not None
            total = relation.messages_opened_30d + relation.messages_dismissed_30d
            user_open_ratio = relation.messages_opened_30d / max(total, 1)
            relation_type = relation.why_user_knows_account
            
        return BusinessContextEvidence(
            is_verified=is_verified,
            has_active_relation=has_active,
            opted_out=opted_out,
            user_open_ratio=user_open_ratio,
            relation_type=relation_type
        )
