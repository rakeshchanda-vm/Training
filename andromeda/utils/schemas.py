from typing import Literal, TypedDict, Optional, List


class CleanText(TypedDict):
    clean_text: str
    relevant_to_target_company: bool


class PlanValidation(TypedDict):
    """Validation of the plan of action."""

    is_valid: bool
    feedback: Optional[str]


class PlanResponse(TypedDict):
    """List of all steps in detail to perform efficient, valuable research."""

    plan_steps: List[str]


class SectionItem(TypedDict):
    section_title: str
    reason: str
    instructions: str
    sub_sections: Optional[List[str]]


class SectionsResponse(TypedDict):
    report_title: str
    sections: List[SectionItem]


class SufficiencyResponse(TypedDict):
    """Response for sufficiency check. Feedback must be constructive, in under 100 words."""

    feedback: Optional[str]
    is_sufficient: bool


class Validation(TypedDict):
    evaluation: str
    completed: bool


class StepMapping(TypedDict):
    plan_step: str


def make_router_schema(options: List, force_routing: bool = False):
    if force_routing:

        class Router(TypedDict):
            """
            message: str = Either a response to the user or a task to the agent
            next: Literal[*options] # type: ignore
            additional_context: Optional[str] = Additional context for the next step
            """

            message: str
            next: Literal[*options]  # type: ignore
            additional_context: Optional[str]

        return Router
    else:

        class Router(TypedDict):
            """
            message: str = Either a response to the user or a task to the agent
            routing_required: bool = Whether the message requires routing to a specialized agent
            next: Literal[*options] # type: ignore
            additional_context: Optional[str] = Additional context for the next step
            """

            message: str
            routing_required: bool
            next: Literal[*options]  # type: ignore
            additional_context: Optional[str]

    return Router
