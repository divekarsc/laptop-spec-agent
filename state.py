from typing import NotRequired, TypedDict

from schema import LaptopSpecs, UseCaseFitEvaluation


class AgentState(TypedDict):
    url: str
    use_case: str
    raw_content: NotRequired[str]
    specs: NotRequired[LaptopSpecs]
    retry_count: NotRequired[int]
    fit_evaluation: NotRequired[UseCaseFitEvaluation]
    error: NotRequired[str]
