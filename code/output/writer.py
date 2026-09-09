import csv
import os
from output.trace import ReasoningTrace

class OutputWriter:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = open(self.filepath, 'w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.file)
        self.writer.writerow(["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"])
        
    def write_row(self, trace: ReasoningTrace):
        ev_ids = trace.evidence_message_ids
        evidence_str = ";".join(ev_ids) if ev_ids else "none"
        self.writer.writerow([
            trace.message_id,
            trace.action,
            trace.message_type,
            trace.reason,
            f"{trace.confidence:.2f}",
            evidence_str
        ])
        
    def flush_and_close(self):
        self.file.flush()
        self.file.close()
