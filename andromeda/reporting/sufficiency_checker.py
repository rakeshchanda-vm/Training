from typing import List
from andromeda import SystemMessage, BaseMessage
from andromeda.utils.prompts import sufficiency_prompt
from andromeda.utils.schemas import SufficiencyResponse
from andromeda.utils.langtils import get_chat_model
from andromeda.config import ModelConfig


def sufficiency_check(
    messages: List[BaseMessage],
    task: str,
    model_config: ModelConfig,
    check_type: str = "preliminary research",
    base_prompt: str = None,
) -> str:
    """
    Checks if a report is sufficient and meets the specified requirements.

    This function uses a language model to evaluate the sufficiency of a report based on the provided messages,
    task description, and check type. It constructs a system prompt using the sufficiency_prompt utility and
    invokes the model with structured output to determine if the report meets the criteria.

    Args:
        messages (List[BaseMessage]): A list of chat messages representing the conversation or report to be checked.
        task (str): A description of the task or objective that the report is expected to address.
        check_type (str, optional): The type of sufficiency check to perform (e.g., "preliminary research").
            Defaults to "preliminary research".
        base_prompt (str, optional): An optional base prompt to further customize the system prompt.
        model_config (ModelConfig, optional): Model configuration for sufficiency checking.
            If None, uses default configuration.

    Returns:
        str: The model's response indicating whether the report is sufficient, typically structured as defined
            by the SufficiencyResponse schema.

    Example:
        response = sufficiency_check(messages, "Analyze market trends", check_type="final review")
    """
    sufficiency_system_prompt = sufficiency_prompt(check_type, task, base_prompt)
    model = get_chat_model(
        model_config=model_config
    )
    response = model.with_structured_output(
        SufficiencyResponse, method="json_schema"
    ).invoke(messages + [SystemMessage(content=sufficiency_system_prompt)])
    return response
