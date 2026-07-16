from __future__ import annotations

import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Literal, Optional

from andromeda import BaseMessage, HumanMessage
from andromeda.config import ModelConfig
from andromeda.config.config import AgentConfig, SupervisorConfig
from andromeda.reporting.mermaid import generate_mermaid_diagram
from andromeda.tools.tools import get_search_context
from andromeda.utils.logger import log_output, log_supervisor
from andromeda.workspace import WorkspacePolicy, WorkspaceSession


class ReportWriter:
    """Supervisor-driven report synthesis over temp-dir research artifacts."""

    def __init__(
        self,
        report_model_config: ModelConfig,
        supervisor_config: SupervisorConfig,
        tmp_dir: Path,
        report_format: Optional[str] = None,
        output_mode: Literal["state", "file", "both"] = "state",
        output_path: Optional[str] = None,
        base_dir: Optional[Path] = None,
        report_agent_count: int = 1,
    ) -> None:
        self.report_model_config = report_model_config
        self.report_format = report_format
        self.output_mode: Literal["state", "file", "both"] = output_mode
        self.output_path: Optional[Path] = Path(output_path).expanduser() if output_path else None
        self.base_dir: Path = Path(base_dir).expanduser() if base_dir is not None else Path.cwd() / "reports"
        self.tmp_dir: Path = Path(tmp_dir).expanduser()
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.final_report_name = "finalReport.md"

        self.workspace_session = WorkspaceSession.create(
            backend="local_fs",
            root=self.tmp_dir,
            policy=WorkspacePolicy(
                read_only=False,
                enable_shell=False,
                tool_profile="full_compatibility",
            ),
        )
        filesystem_tools = self.workspace_session.tools()

        report_agents: List[AgentConfig] = []
        report_agents.append(
            AgentConfig(
                name=f"report_writer",
                model=self.report_model_config,
                tools=list(list(filesystem_tools.values())),
                recursion_limit=100,
                prompt=dedent(
                    f"""
                    You are a report writing specialist working only inside: {self.tmp_dir}

                    Rules:
                    - Use only local markdown findings already produced in this directory.
                    - Do not browse externally or invent unsupported facts.
                    - Read all relevant findings needed to produce a complete, professional section.
                    - Write section to the final report file in this directory. No other files should be created.
                    - Keep content factual, structured, and directly tied to available findings.
                    - Avoid generic or high-level-only prose; include concrete evidence, specifics, and implications.
                    - Do not mention the tmp files, directory or the final report file in your response.
                    - Preserve and use source citations exactly as [Search #<number>] from the research notes.
                    - Never convert [Search #<number>] into another numbering scheme.
                    - Never mention any temporary filenames (for example *.md files) or the existence of artifacts in the report text.
                    - Use good markdown formatting and structure - ## for headings and ### for subheadings, bullets, tables, etc as needed.
                    - Use mermaid js diagrams to illustrate the findings if applicable.
                    - Use generic, professional figure headings/captions; never include the word "Mermaid" in report headings or captions.
                    - Use markdown tables to illustrate the findings if applicable.
                    - Ensure the report is thorough and detailed.
                    - Synthesize each section directly into final conclusions and insights; do not mirror intermediate analysis templates.
                    - Do not use process/meta labels such as "Objective", "Methodology", "Evidence", "Approach", or "Steps" unless explicitly requested by the user.
                    - If evidence is missing for a claim, remove the claim unless you can find evidence elsewhere..
                    - Adhere strictly to the provided report format and section sequence.
                    - If a section lacks enough evidence, keep the section and write "Insufficient Evidence" with precise missing-data bullets; do not invent facts. Ensure you have fully searched for relevant facts before writing the section.
                    - Include action plans/recommendations only in the single section designated for account strategy or action plan IF requested by the user in the required format.
                    - Do not add "recommended actions", "next steps", or equivalent action lists in any other section.
                    - Use the exact [Search #<number>] tags in the report text. Do not include any other provenance or references. Do not override this rule.
                    - Do not include any kind of notes or commentary about the report or the findings in the report text. Do not override this rule.
                    - Do not include any parenthesis in the report headings. 

                    What you write will be the professional report used as is. Use only write_file tool when the file 
                    does not exist. When file exists with prior content, use edit_file or append_to_file tool to update the file.
                    As a prerequisite, you need to read the final report file to understand the context and structure of the report.

                    Keep your context limitations in mind and do not exceed them.

                    Final report file: {self.final_report_name}
                    """
                ).strip(),
            )
        )

        report_supervisor_config = supervisor_config.model_copy(deep=False)
        report_supervisor_config.name = "report_supervisor"
        report_supervisor_config.enable_planning = True
        report_supervisor_config.prompt = (
            (report_supervisor_config.prompt or "")
            + "\n\n"
            + dedent(
                f"""
                You are now acting as the report supervisor.

                Workspace:
                - Directory: {self.tmp_dir}
                - Final output file: {self.final_report_name}

                Workflow requirements:
                - Inspect directory and read all relevant findings needed for full coverage.
                - Assign report sections to sub-agents using route_to_agent.
                - Instruct sub-agents to produce section markdown content assembled into a single final markdown report: {self.final_report_name}.
                - Include format requirements, structure and recommended markdown formatting and mermaid diagrams when applicable.
                - Ensure final report is coherent and references concrete findings.
                - Ensure citations remain in [Search #<number>] format across the full report.
                - Never allow temporary markdown filenames or the existence of artifacts to appear in final prose.
                - Reject and rewrite sections that are generic, unsupported, or too shallow.
                - Reject and rewrite sections that read like process notes or intermediate analysis checklists.
                - Enforce strict report format compliance; if a section lacks evidence, keep the section and mark it as "Insufficient Evidence" with concrete data gaps.
                - Enforce that action-plan bullets appear in exactly one designated section only.
                - Review final report when done and assign additional tasks if required to clean it up.

                IMPORTANT: Each sub agent MUST be given either a small section or a sub section of the report to write, not the entire report.
                It is your duty to either use your tools or another agent to ensure the report structure is maintained, consistent and complete.

                Report requirements:
                - The first agent should write the heading for the report along with it's relevant section.
                - Grounded in the findings available in the directory.
                - The report should be a single markdown file with a clear professional structure and content.
                - The report can contain mermaid diagrams to illustrate the findings. Proactively suggest agents to use them when appropriate.
                - Diagram section headings/captions must be generic and professional; never label a heading as "Mermaid diagram ...".
                - Proper headings and subheadings should be used to structure the report.
                - Consistent markdown formatting and structure, ## for headings and ### for subheadings, bullets, tables, etc as needed.
                - Thorough and detailed section writing.
                - Keep section prose direct and synthesis-first; avoid meta commentary about how the analysis was conducted unless user explicitly requests it.
                - Custom document uploads may not have URLs; preserve their [Search #<number>] citations without inventing URLs.
                """
            ).strip()
        )

        current_tools = list(report_supervisor_config.tools)
        report_supervisor_config.tools = list(current_tools) + list(list(filesystem_tools.values()))

        from andromeda.core.supervisor import Supervisor

        self.supervisor = Supervisor(
            agents=report_agents,
            config=report_supervisor_config,
        )

    def _resolve_output_path(self) -> Path:
        if self.output_path is not None:
            target = self.output_path
        else:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            target = self.base_dir / f"report-{timestamp}.md"

        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def _strip_think(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        if "</think>" in text:
            return text.split("</think>", 1)[1].strip()
        return text.strip()

    def _latest_message_text(self, messages: List[BaseMessage]) -> str:
        if not messages:
            return ""
        content = getattr(messages[-1], "content", "")
        if isinstance(content, str):
            return self._strip_think(content)
        return str(content)

    def _ensure_final_report_exists(self, fallback_text: str) -> Path:
        final_path = self.tmp_dir / self.final_report_name
        if final_path.exists():
            return final_path

        log_supervisor(
            f"{self.final_report_name} was not created by report supervisor; writing fallback report"
        )
        fallback_content = fallback_text or "# Final Report\n\nNo report content was generated."
        final_path.write_text(fallback_content, encoding="utf-8")
        return final_path

    def _build_supervisor_prompt(self, state: Dict[str, Any]) -> str:
        messages = state.get("messages", [])
        original_task = ""
        if messages:
            first = getattr(messages[0], "content", "")
            original_task = first if isinstance(first, str) else str(first)

        extra_format = self.report_format.strip() if isinstance(self.report_format, str) else ""

        return dedent(f"""
            Create the final report from research artifacts in {self.tmp_dir}.

            Task:
            {original_task}

            Workflow:
            - Inspect research files sufficiently to cover all major findings and evidence
            - Assign section drafting tasks to report sub-agents, in the same file
            - Require each section to include grounded evidence, analysis, and professional formatting
            - Finalize only after checking report completeness and consistency

            Constraints:
            - Ensure section content is grounded in available findings.
            - Strictly no assumed or hallucinated information.
            - Preserve source tags exactly as [Search #<number>] wherever evidence is used.
            - Never mention temporary markdown filenames inside the report.
            - Diagram headings/captions must be generic and must not specify "Mermaid".
            - Write direct synthesized section content, not process-oriented analysis notes.
            - Do not include headings/labels like "Objective", "Methodology", "Evidence", or "Approach" unless explicitly requested.
            - Follow the provided report format strictly and keep section order unchanged.
            - If evidence is unavailable (after thorough search) for a required section, write "Insufficient Evidence" plus concrete missing-data bullets in that section.
            - Ensure action-plan recommendations appear only in the single designated action-plan/account-strategy section.
            - Do not include any parenthesis in the report headings. Do not include references apart from [Search #<number>] in the report text.
            - Return a short completion note confirming report completion.
            {f'- Follow this report format strictly: {extra_format}' if extra_format else 'Not provided'}
            """
        ).strip()

    def _render_mermaid_if_present(self, report_path: Path) -> str:
        text = report_path.read_text(encoding="utf-8")
        if "```mermaid" not in text.lower():
            return text

        assets_dir = report_path.parent / "assets"
        rendered = generate_mermaid_diagram(
            text=text,
            output_path=assets_dir,
            fname=report_path.stem,
        )
        report_path.write_text(rendered, encoding="utf-8")
        return rendered

    def _remove_tmp_filename_mentions(self, text: str) -> str:
        cleaned = text
        candidate_files = [p.name for p in self.tmp_dir.glob("*.md") if p.name != self.final_report_name]
        candidate_files.append(self.final_report_name)
        for name in sorted(set(candidate_files), key=len, reverse=True):
            cleaned = re.sub(rf"`{re.escape(name)}`", "research notes", cleaned)
            cleaned = re.sub(rf"\b{re.escape(name)}\b", "research notes", cleaned)
        return cleaned

    def _normalize_diagram_headings(self, text: str) -> str:
        normalized = re.sub(
            r"^(#{1,6}\s*)Mermaid\s+diagram\s*[-:|]\s*(.+?)\s*$",
            r"\1\2",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        normalized = re.sub(
            r"^(#{1,6}\s*)Mermaid\s+diagram\s*$",
            r"\1Diagram",
            normalized,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        return normalized

    def _build_source_map_section(self, report_text: str) -> str:
        used_ids = sorted({int(match) for match in re.findall(r"\[Search #(\d+)\]", report_text)})
        all_sources = get_search_context() or []
        source_by_id = {int(s.get("id")): s for s in all_sources if isinstance(s, dict) and s.get("id") is not None}

        lines: List[str] = ["\n\n## Source Map\n"]
        if not used_ids:
            lines.append("- No [Search #<number>] citations were found in the report.\n")
            return "".join(lines)

        for source_id in used_ids:
            source = source_by_id.get(source_id)
            if not source:
                lines.append(f"- [Search #{source_id}] (source metadata not found)\n")
                continue
            data = source.get("data", {}) if isinstance(source.get("data", {}), dict) else {}
            title = str(data.get("title", "Untitled")).strip() or "Untitled"
            url = str(data.get("url", "")).strip()
            query = str(source.get("query", "")).split("<<<", 1)[0].strip()
            if query:
                lines.append(f"- [Search #{source_id}] {title} ({url}) | query: {query}\n")
            else:
                lines.append(f"- [Search #{source_id}] {title} ({url})\n")
        return "".join(lines)

    def _finalize_report_text(self, report_text: str) -> str:
        cleaned = self._remove_tmp_filename_mentions(report_text)
        cleaned = self._normalize_diagram_headings(cleaned)
        source_map = self._build_source_map_section(cleaned)
        without_existing = re.sub(r"\n## Source Map\s*[\s\S]*$", "", cleaned, flags=re.IGNORECASE).rstrip()
        return without_existing + source_map

    def report_generator(self, state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._build_supervisor_prompt(state)

        report_state: Dict[str, Any] = {
            "messages": [HumanMessage(content=prompt)],
            "plan": [
                "Inspect tmp research files minimally",
                "Assign section drafting tasks to report sub-agents",
                "Assemble finalReport.md",
            ],
        }

        supervisor_result = self.supervisor.supervise(report_state)
        supervisor_messages = supervisor_result.get("messages", []) if isinstance(supervisor_result, dict) else []
        fallback_text = self._latest_message_text(supervisor_messages)

        tmp_report_path = self._ensure_final_report_exists(fallback_text)
        result: Dict[str, Any] = {}

        if self.output_mode == "state":
            full_report = self._render_mermaid_if_present(tmp_report_path)
            full_report = self._finalize_report_text(full_report)
            tmp_report_path.write_text(full_report, encoding="utf-8")
            result["report_output"] = full_report
            try:
                tmp_report_path.unlink(missing_ok=True)
            except Exception as exc:  # pragma: no cover
                log_supervisor(f"Failed to delete temporary report {tmp_report_path}: {exc}")
        else:
            destination = self._resolve_output_path()
            src_resolved = tmp_report_path.resolve()
            dst_resolved = destination.resolve()
            if src_resolved != dst_resolved:
                if destination.exists():
                    destination.unlink()
                shutil.move(str(tmp_report_path), str(destination))
            full_report = self._render_mermaid_if_present(destination)
            full_report = self._finalize_report_text(full_report)
            destination.write_text(full_report, encoding="utf-8")
            result["report_file_path"] = str(destination)
            if self.output_mode == "both":
                result["report_output"] = full_report

        log_output("Final report generated successfully.")
        return result
