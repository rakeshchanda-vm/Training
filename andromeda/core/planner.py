from typing import Any, Dict, List, Optional, Union
from andromeda.utils.langtils import BaseChatModel
from andromeda import HumanMessage, AIMessage

from andromeda.config.config import PlannerConfig
from andromeda.utils.langtils import get_chat_model
from andromeda.utils.prompts import plan_validation_prompt, plan_of_action_prompt
from andromeda.utils.schemas import PlanValidation, PlanResponse


class PlannerAgent:
    """Agent responsible for generating and validating plans of action.

    The PlannerAgent class encapsulates the logic for generating a plan of action
    using a language model and validating the generated plan. It leverages configuration
    options to initialize the underlying chat model and supports structured output
    validation using JSON schemas. The agent can interact with multiple sub-agents
    and is designed to be extensible for various planning workflows.

    Attributes:
        config (PlannerConfig): Configuration object for the planner, including model and report settings.
        model: The initialized chat model used for generating and validating plans.
        report_format: The structure or format in which reports/plans are generated.
    """

    def __init__(
        self,
        config: Optional[PlannerConfig] = None,
        agents: Union[Dict[str, Any], List[Any]] = [],
        plan: str | List = [],
    ):
        """Initializes the PlannerAgent with an optional configuration.

        If no configuration is provided, a default PlannerConfig is used. The chat model
        is initialized based on the configuration, and the report format is set accordingly.

        Args:
            config (Optional[PlannerConfig]): Optional configuration object for the planner.
            agents (Dict[str, Agent]): A dictionary of available agents, keyed by agent name.
            plan (str | List, optional): An initial plan or list of steps to seed the planning process.
        """
        self.config = config or PlannerConfig()
        self.model: BaseChatModel = get_chat_model(model_config=self.config.model)
        self.report_format = self.config.report_structure
        self.agents = agents
        self.plan = plan

    def validate_plan(self, state, plan: list) -> bool:
        """Validates a proposed plan of action using the language model.

        This method constructs a validation prompt and uses the chat model to
        check if the provided plan meets the required criteria. The validation
        is performed using a structured output schema (PlanValidation) and
        returns the validation result, which includes whether the plan is valid
        and any feedback for improvement.

        Args:
            state (dict): The current state, including the conversation messages.
            plan (list): The plan of action to be validated, as a list of steps.

        Returns:
            dict: The validation result, including 'is_valid' (bool) and 'feedback' (str).
        """
        validation_prompt = plan_validation_prompt
        validation = self.model.with_structured_output(
            PlanValidation, method="json_schema"
        ).invoke(
            state["messages"]
            + [
                AIMessage(
                    content="Plan of action:\n" + "\n".join(plan), name="assistant"
                ),
                HumanMessage(content=validation_prompt),
            ]
        )
        return validation

    def plan_of_action(self, state: Dict[str, Any]) -> Dict[str, List]:
        """Generates a plan of action based on the current state and available agents.

        This method constructs a prompt describing the available agents and their tools,
        then iteratively generates and validates a plan of action using the chat model.
        If the generated plan is invalid or insufficient, the process repeats with feedback
        until a valid plan is produced. The final plan is logged and returned as a list of steps.

        Args:
            state (Dict[str, Any]): The current state, including the conversation messages.

        Returns:
            Dict[str, List]: The validated plan of action as a list of steps.
        """
        plan_prompt = plan_of_action_prompt(
            agent_details=", ".join(
                list(
                    set(
                        [
                            f"{agent.name}: {agent.tools}"
                            for agent in self.agents.values()
                        ]
                    )
                )
                if isinstance(self.agents, dict)
                else list(
                    set([f"{agent.name}: {agent.tools}" for agent in self.agents])
                )
            ),
            plan=self.plan,
            report_format=self.report_format,
            task_type=getattr(self.config, "task_type", "general"),
        )
        max_attempts = 3
        attempts = 0
        plan = None
        while attempts < max_attempts:
            attempts += 1
            plan = self.model.with_structured_output(
                PlanResponse, method="json_schema"
            ).invoke(
                state["messages"] + [HumanMessage(content=plan_prompt, name="human")]
            )
            plan = plan["plan_steps"]
            if not plan or len(plan) < 2:
                # log_output("Could not generate a plan of action. Trying again")
                continue
            validation = self.validate_plan(state, plan)

            if validation["is_valid"]:
                break
            else:
                plan_prompt += "Feedback: " + validation["feedback"]

        # use the last generated plan anyway if we have exhausted all attempts

        return {"plan": plan}
