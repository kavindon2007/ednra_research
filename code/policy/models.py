from dataclasses import dataclass, field
from typing import List

@dataclass(frozen=True)
class PolicyDecision:
    """
    A decision rendered by the PolicyEngine.
    """
    policy_name: str
    explanation: str
    evidence_used: List[str] = field(default_factory=list)
    priority: int = 0
