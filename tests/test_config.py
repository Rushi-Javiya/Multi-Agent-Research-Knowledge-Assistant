import pytest
from pydantic import ValidationError

from research_assistant.config import Settings


def test_defaults_are_bounded():
    s = Settings(_env_file=None)
    assert s.max_revisions == 2
    assert s.require_human_review is True


def test_env_override(monkeypatch):
    monkeypatch.setenv("RESEARCH_MAX_REVISIONS", "4")
    assert Settings(_env_file=None).max_revisions == 4


def test_unbounded_revisions_rejected(monkeypatch):
    monkeypatch.setenv("RESEARCH_MAX_REVISIONS", "1000")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
