from typing import Literal

from pydantic import Field

from .base import MADNEModel

IssueCategory = Literal["logic", "coherence", "literary", "character"]
Severity = Literal["low", "medium", "high"]


class Issue(MADNEModel):
    category: IssueCategory
    detail: str
    severity: Severity = "medium"


class ReviewReport(MADNEModel):
    """GuardianAI 的审查报告。"""

    passed: bool
    issues: list[Issue] = Field(default_factory=list)
    suggested_fix: str | None = None
