import urllib.request
import json
import logging
from typing import Optional

from context.models import IncomingMessage, LLMClassifierEvidence

logger = logging.getLogger("orchestrator")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:latest"

PROMPT_TEMPLATE = """
You are a Semantic Analyzer for an intelligent messaging system.
Your job is ONLY to extract semantic meaning from incoming messages.
You MUST NOT make routing decisions (like notify, digest, mute).

Message Text:
{text}

Media Context (if any):
{media}

Analyze the message and return ONLY a valid JSON object matching this schema:
{{
  "message_type": "<choose exactly one: personal, group_chat, promotion, transactional, otp, scam, spam, reminder, other>",
  "semantic_urgency": <float between 0.0 and 1.0 representing how urgent the semantic meaning is>,
  "model_reasoning": "<brief explanation of why this type and urgency were chosen>"
}}

Respond with ONLY the JSON object. No markdown formatting, no backticks, no introductory text.
"""

class LLMClassifier:
    @classmethod
    def classify(cls, message: IncomingMessage, media_desc: str = "") -> LLMClassifierEvidence:
        prompt = PROMPT_TEMPLATE.format(
            text=message.message_text,
            media=media_desc if media_desc else "None"
        )
        
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.0
            }
        }
        
        req = urllib.request.Request(OLLAMA_URL, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
        
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode('utf-8'))
                llm_output = result.get("response", "").strip()
                
                # Clean up potential markdown formatting from Ollama
                if llm_output.startswith("```json"):
                    llm_output = llm_output[7:]
                if llm_output.endswith("```"):
                    llm_output = llm_output[:-3]
                    
                parsed = json.loads(llm_output.strip())
                
                return LLMClassifierEvidence(
                    message_type=parsed.get("message_type", "unknown"),
                    semantic_urgency=float(parsed.get("semantic_urgency", 0.0)),
                    model_reasoning=parsed.get("model_reasoning", ""),
                    used_llm=True
                )
        except Exception as e:
            logger.warning(f"LLM classification failed for msg {message.message_id}: {e}")
            return LLMClassifierEvidence(
                message_type="unknown",
                semantic_urgency=0.0,
                model_reasoning=f"Error: {e}",
                used_llm=False
            )
