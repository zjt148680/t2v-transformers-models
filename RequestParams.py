from typing import List, Optional

from pydantic import BaseModel


class VectorInputConfig(BaseModel):
    pooling_strategy: Optional[str] = None
    task_type: Optional[str] = None

    def __hash__(self):
        return hash((self.pooling_strategy, self.task_type))

    def __eq__(self, other):
        if isinstance(other, VectorInputConfig):
            return (
                    self.pooling_strategy == other.pooling_strategy
                    and self.task_type == other.task_type
            )
        return False


class VectorInput(BaseModel):
    text: str
    config: Optional[VectorInputConfig] = None

    def __hash__(self):
        return hash((self.text, self.config))

    def __eq__(self, other):
        if isinstance(other, VectorInput):
            return self.text == other.text and self.config == other.config
        return False


class BatchVectorInput(BaseModel):
    texts: List[str]
    config: Optional[VectorInputConfig] = None

    def __hash__(self):
        return hash((self.texts, self.config))

    def __eq__(self, other):
        if isinstance(other, BatchVectorInput):
            return self.texts == other.texts and self.config == other.config
        return False
