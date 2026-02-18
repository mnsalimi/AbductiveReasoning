from ..metric import AbductiveMetric
from typing import Dict, Any
import yaml
from pydantic import BaseModel, Field

class BranchinessResponse(BaseModel):
    list_of_branches: str = Field(
        description=""
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1"
    )

class Branchiness(AbductiveMetric):
    def construct_prompt(self, dataset_name: str, sample: Dict[str, Any])->tuple[str, str, BaseModel]:
        with open(f"prompts.yaml", "r") as f:
            prompts_config = yaml.safe_load(f)
        system_prompt = prompts_config['metric_system_prompts']['branchiness'][dataset_name]["branchiness"]["prompt"]
        user_prompt = ""
        basemodel = BranchinessResponse
        return system_prompt, user_prompt, basemodel

    def parse_response(self, dataset_name: str, response: str)->dict:
        pass

    def __str__(self) -> str:
        return "branchiness"