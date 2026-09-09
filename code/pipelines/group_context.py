from pipelines.base import EvidencePipeline
from context.models import IncomingMessage, GroupContextEvidence
from context.loader import DataContext

class GroupContextPipeline(EvidencePipeline[GroupContextEvidence | None]):
    def run(self, message: IncomingMessage, context: DataContext) -> GroupContextEvidence | None:
        if not message.group_id:
            return None
            
        group = context.groups.get(message.group_id)
        if not group:
            return None
            
        membership = context.group_members.get((message.group_id, message.user_id))
        if not membership:
            return None
            
        reads = max(membership.messages_read_30d, 1)
        engagement = membership.replies_sent_30d / reads
        dismiss_rate = membership.notifications_dismissed_30d / reads
        
        sender_membership = context.group_members.get((message.group_id, message.sender_user_id)) if message.sender_user_id else None
        sender_is_admin = sender_membership.role == "admin" if sender_membership else False
        
        return GroupContextEvidence(
            group_type=group.group_type,
            is_muted_by_user=membership.group_muted_by_user,
            user_engagement_ratio=engagement,
            user_dismiss_rate_in_group=dismiss_rate,
            sender_is_admin=sender_is_admin
        )
