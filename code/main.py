import os
import sys
import logging

from context.loader import DataContext
from context.models import EvidenceBundle

from pipelines.scam_risk import ScamRiskPipeline
from pipelines.sender_trust import SenderTrustPipeline
from pipelines.user_behavior import UserBehaviorPipeline
from pipelines.group_context import GroupContextPipeline
from pipelines.business_context import BusinessContextPipeline
from pipelines.content_signals import ContentSignalsPipeline
from pipelines.media_signals import MediaSignalsPipeline
from pipelines.historical_match import HistoricalMatchPipeline

from inference.llm_classifier import LLMClassifier
from inference.engine import InferenceEngine
from policy.engine import PolicyEngine
from policy.dsi import DSI

from output.writer import OutputWriter
from output.trace_builder import TraceBuilder
from output.trace import ReasoningTrace

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
logger = logging.getLogger("orchestrator")

def main():
    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
    output_csv = os.path.join(dataset_dir, "output.csv")
    
    if not os.path.exists(dataset_dir):
        logger.critical(f"Dataset directory not found: {dataset_dir}")
        sys.exit(1)
        
    logger.info("Loading context...")
    ctx = DataContext(dataset_dir)
    logger.info(f"Context loaded: {len(ctx.messages)} messages")
    
    writer = OutputWriter(output_csv)
    
    # Initialize pipelines
    p_scam = ScamRiskPipeline()
    p_trust = SenderTrustPipeline()
    p_behavior = UserBehaviorPipeline()
    p_group = GroupContextPipeline()
    p_biz = BusinessContextPipeline()
    p_content = ContentSignalsPipeline()
    p_media = MediaSignalsPipeline()
    p_history = HistoricalMatchPipeline()
    
    trace_builder = TraceBuilder()
    
    counts = {"notify": 0, "digest": 0, "mute": 0}
    
    logger.info("Starting processing...")
    
    for i, msg in enumerate(ctx.messages):
        try:
            scam = p_scam.run(msg, ctx)
            trust = p_trust.run(msg, ctx)
            behavior = p_behavior.run(msg, ctx)
            group = p_group.run(msg, ctx)
            biz = p_biz.run(msg, ctx)
            content = p_content.run(msg, ctx)
            media = p_media.run(msg, ctx)
            history = p_history.run(msg, ctx)
            
            llm = LLMClassifier.classify(msg, media.media_description)
            
            bundle = EvidenceBundle(
                message=msg,
                scam_risk=scam,
                sender_trust=trust,
                user_behavior=behavior,
                content_signals=content,
                media_signals=media,
                historical_match=history,
                llm_classifier=llm,
                group_context=group,
                business_context=biz
            )
            
            state = InferenceEngine.infer(bundle)
            action, decision = PolicyEngine.decide(state, bundle)
            confidence = DSI.compute(state, bundle, action)
            
            msg_type = bundle.llm_classifier.message_type if bundle.llm_classifier.used_llm else "unknown"
            if not bundle.llm_classifier.used_llm:
                if bundle.content_signals.has_payment_keyword or bundle.content_signals.has_event_keyword:
                    msg_type = "transactional"
                elif bundle.content_signals.is_forward_chain:
                    msg_type = "spam"
                    
            trace = trace_builder.build(
                bundle=bundle,
                state=state,
                decision=decision,
                action=action,
                message_type=msg_type,
                confidence=confidence
            )
            
            writer.write_row(trace)
            
            counts[trace.action] = counts.get(trace.action, 0) + 1
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1} messages...")
                
        except Exception as e:
            logger.error(f"Failed to process message {msg.message_id}: {e}", exc_info=True)
            
    writer.flush_and_close()
    
    total = sum(counts.values())
    logger.info(f"✓ Complete: {total}/{len(ctx.messages)} | notify={counts['notify']} digest={counts['digest']} mute={counts['mute']}")

if __name__ == "__main__":
    main()
