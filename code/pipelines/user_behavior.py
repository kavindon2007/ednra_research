from pipelines.base import EvidencePipeline
from context.models import IncomingMessage, UserBehaviorEvidence
from context.loader import DataContext
from datetime import datetime

class UserBehaviorPipeline(EvidencePipeline[UserBehaviorEvidence]):
    def run(self, message: IncomingMessage, context: DataContext) -> UserBehaviorEvidence:
        user = context.users.get(message.user_id)
        if not user:
            return UserBehaviorEvidence(in_dnd=False, dismiss_rate=0.0, report_rate=0.0, fatigue_score=0.0)
            
        in_dnd = False
        if user.do_not_disturb_window and "-" in user.do_not_disturb_window:
            start_str, end_str = user.do_not_disturb_window.split("-")
            try:
                msg_time = datetime.fromisoformat(message.created_at).time()
                start = datetime.strptime(start_str, "%H:%M").time()
                end = datetime.strptime(end_str, "%H:%M").time()
                if start <= end:
                    in_dnd = start <= msg_time <= end
                else:
                    in_dnd = msg_time >= start or msg_time <= end
            except ValueError:
                pass
                
        dismiss_rate = user.notifications_dismissed_30d / user.messages_opened_30d if user.messages_opened_30d > 0 else 0.0
        report_rate = user.messages_reported_30d / user.messages_opened_30d if user.messages_opened_30d > 0 else 0.0
        
        # Simple fatigue score: average dismiss/sent from daily summary
        fatigue_score = 0.0
        summaries = context.daily_summaries.get(message.user_id, [])
        if summaries:
            total_sent = sum(s.notifications_sent for s in summaries)
            total_dismiss = sum(s.notifications_dismissed for s in summaries)
            fatigue_score = total_dismiss / total_sent if total_sent > 0 else 0.0
            
        return UserBehaviorEvidence(
            in_dnd=in_dnd,
            dismiss_rate=dismiss_rate,
            report_rate=report_rate,
            fatigue_score=fatigue_score
        )
