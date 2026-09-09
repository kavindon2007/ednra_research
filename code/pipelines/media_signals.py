from pipelines.base import EvidencePipeline
from context.models import IncomingMessage, MediaSignalsEvidence
from context.loader import DataContext
import os

class MediaSignalsPipeline(EvidencePipeline[MediaSignalsEvidence]):
    def run(self, message: IncomingMessage, context: DataContext) -> MediaSignalsEvidence:
        has_media = message.media_type in ("image", "voice")
        media_desc = ""
        urgency = 0.0
        
        if has_media and message.media_id:
            # We don't have a local multimodal model that works fast enough to process all images
            # or audio deterministically in this step, but we simulate extraction or use file paths.
            if message.media_type == "image":
                record = context.images.get(message.media_id)
                if record:
                    media_desc = f"[Image Attached: {os.path.basename(record.file_path)}]"
            elif message.media_type == "voice":
                record = context.voice_notes.get(message.media_id)
                if record:
                    media_desc = f"[Voice Note Attached: {os.path.basename(record.file_path)}]"
                    
        return MediaSignalsEvidence(
            has_media=has_media,
            media_type=message.media_type if has_media else "",
            media_id=message.media_id if has_media else "",
            media_description=media_desc,
            inferred_urgency=urgency
        )
