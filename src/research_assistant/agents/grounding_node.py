"""Graph node that gates every draft behind the deterministic grounding check."""

from typing import Any

from research_assistant.agents.common import NodeFn
from research_assistant.grounding import feedback_for_writer, render_report, validate_grounding
from research_assistant.logging_config import get_logger
from research_assistant.models import ResearchStatus, ReviewDecision, ReviewVerdict
from research_assistant.state import ResearchState

log = get_logger("grounding")


def make_grounding_node(max_revisions: int, min_coverage: float) -> NodeFn:
    def grounding_node(state: ResearchState) -> dict[str, Any]:
        sources = state.get("sources", [])
        result = validate_grounding(state.get("draft", ""), sources, min_coverage)

        if result.ok:
            return {
                "grounding": result,
                "report": render_report(state["draft"], sources, result.cited_ids),
                "review": None,
                "status": ResearchStatus.REVIEWING,
            }

        log.warning("grounding rejected the draft: %s", "; ".join(result.problems))
        if state.get("revision_count", 0) >= max_revisions:
            return {
                "grounding": result,
                "report": "",
                "status": ResearchStatus.FAILED,
                "errors": [
                    f"no grounded report after {max_revisions} revision(s): "
                    + "; ".join(result.problems)
                ],
            }
        return {
            "grounding": result,
            "report": "",
            "review": ReviewVerdict(
                decision=ReviewDecision.REVISE, feedback=feedback_for_writer(result, sources)
            ),
        }

    return grounding_node
