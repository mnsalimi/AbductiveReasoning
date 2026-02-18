from abc import ABC, abstractmethod
from typing import Dict, Any
from pydantic import BaseModel

class AbductiveMetric(ABC):
    """
    Abstract class for abductive metrics.
    """
    @abstractmethod
    def construct_prompt(self, dataset_name: str, sample: Dict[str, Any])->tuple[str, str, BaseModel]:
        """
        Construct a prompt for the metric.
        """
        pass

    @abstractmethod
    def parse_response(self, dataset_name: str, response: str)->dict:
        """
        Parse the response from the model.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass