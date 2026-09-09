"""
pipelines/base.py

Abstract base class for all evidence pipelines.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from context.models import Evidence, IncomingMessage
from context.loader import DataContext

T = TypeVar('T', bound=Evidence)


class EvidencePipeline(ABC, Generic[T]):
    """
    Base class for pipelines that extract structured evidence from
    an incoming message and the surrounding data context.
    """

    @abstractmethod
    def run(self, message: IncomingMessage, context: DataContext) -> T:
        """
        Extract and return evidence. 
        Must be deterministic and raise exceptions only on fatal data errors.
        """
        pass
