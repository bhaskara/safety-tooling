"""The Anthropic adapter returns an empty completion (not an IndexError) when the API sends a message with no content blocks."""

from types import SimpleNamespace

import pytest

from safetytooling.apis.inference.anthropic import AnthropicChatModel
from safetytooling.data_models import ChatMessage, MessageRole, Prompt


class _FakeMessages:
    """Stand-in for ``AsyncAnthropic().messages`` whose ``create`` returns a message with no content blocks."""

    def __init__(self, stop_reason, content=()):
        self.stop_reason = stop_reason
        self.content = list(content)
        self.calls = 0

    async def create(self, **_kwargs):
        self.calls += 1
        return SimpleNamespace(content=list(self.content), stop_reason=self.stop_reason)


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_reason, expected", [("refusal", "content_filter"), (None, "unknown")])
async def test_empty_content_list_becomes_empty_completion(stop_reason, expected):
    """One API call, no retry, completion "" and the normalised stop reason (``refusal`` -> content_filter; none sent -> unknown)."""
    model = AnthropicChatModel(num_threads=1, anthropic_api_key="test-key-not-used")
    fake = _FakeMessages(stop_reason)
    model.aclient = SimpleNamespace(messages=fake)
    prompt = Prompt(messages=[ChatMessage(role=MessageRole.user, content="judge this")])
    responses = await model(model_id="claude-sonnet-4-5", prompt=prompt, print_prompt_and_response=False, max_attempts=3)
    assert fake.calls == 1
    assert len(responses) == 1
    assert responses[0].completion == ""
    assert str(responses[0].stop_reason) == expected


@pytest.mark.asyncio
async def test_thinking_without_text_block_becomes_empty_completion():
    """A response holding only a thinking block (no text block) yields completion "" and keeps the thinking as reasoning_content."""
    model = AnthropicChatModel(num_threads=1, anthropic_api_key="test-key-not-used")
    fake = _FakeMessages("end_turn", content=[SimpleNamespace(type="thinking", thinking="hmm")])
    model.aclient = SimpleNamespace(messages=fake)
    prompt = Prompt(messages=[ChatMessage(role=MessageRole.user, content="judge this")])
    responses = await model(model_id="claude-sonnet-4-5", prompt=prompt, print_prompt_and_response=False, max_attempts=3)
    assert fake.calls == 1 and len(responses) == 1
    assert responses[0].completion == ""
    assert str(responses[0].stop_reason) == "stop_sequence"
    assert responses[0].reasoning_content == "hmm"
