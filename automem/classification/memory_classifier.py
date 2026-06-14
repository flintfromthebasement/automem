from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional


class MemoryClassifier:
    """Classifies memories into specific types based on content patterns."""

    MAX_COMPLETION_PREFIXES = ("o1", "o3", "o4", "gpt-4o", "gpt-4.1", "gpt-5")

    PATTERNS = {
        "Decision": [
            r"decided to",
            r"chose (\w+) over",
            r"going with",
            r"picked",
            r"selected",
            r"will use",
            r"choosing",
            r"opted for",
        ],
        "Pattern": [
            r"usually",
            r"typically",
            r"tend to",
            r"pattern i noticed",
            r"often",
            r"frequently",
            r"regularly",
            r"consistently",
        ],
        "Preference": [
            r"prefer",
            r"like.*better",
            r"favorite",
            r"always use",
            r"rather than",
            r"instead of",
            r"favor",
        ],
        "Style": [
            r"wrote.*in.*style",
            r"communicated",
            r"responded to",
            r"formatted as",
            r"using.*tone",
            r"expressed as",
        ],
        "Habit": [
            r"\balways\b(?!\s+use\b)",
            r"every time",
            r"habitually",
            r"routine",
            r"daily",
            r"weekly",
            r"monthly",
        ],
        "Insight": [
            r"realized",
            r"discovered",
            r"learned that",
            r"understood",
            r"figured out",
            r"insight",
            r"revelation",
        ],
        "Context": [
            r"during",
            r"while working on",
            r"in the context of",
            r"when",
            r"at the time",
            r"situation was",
        ],
    }

    SYSTEM_PROMPT = """You classify a single memory into exactly ONE type. Read the content carefully and pick the most specific type that fits. When multiple fit, use the priority rules at the bottom.

TYPES (with strict definitions):

- **Decision**: A choice was actively made between alternatives, or a commitment was made. Keywords: "chose", "decided", "will use", "picked X over Y", "going with". NOT every recommendation or protocol is a Decision — only if *someone made a choice*.

- **Pattern**: A reusable approach, template, or recurring way of doing something. Keywords: "pattern", "approach", "template", "workflow", "always do X when Y". Repeatable and generalizable.

- **Preference**: A stated like/dislike, favorite, or taste. Someone prefers X. Keywords: "prefers", "likes", "dislikes", "favorite", "wants X over Y" (as a taste, not a decision).

- **Style**: Formatting, tone, naming convention, or communication approach. Keywords: "tabs not spaces", "short commit messages", "formal tone", "snake_case".

- **Habit**: A regular routine or repeated behavior. Keywords: "always X", "every morning", "routinely", "every Friday". Time-regularity is the marker.

- **Insight**: A genuine *realization* or *learning* from experience — something the author *discovered* or *figured out*. Keywords: "turns out", "learned that", "root cause", "realized", "the trick is", "gotcha". NOT textbook facts or product descriptions.

- **Context**: Background facts about a person, place, product, tool, or situation. Keywords: "X is a Y", "X offers Y", "X released", "X is located at", biographical facts, product descriptions, tool capabilities, session logs, conversation fragments. This is the correct type for factual statements that aren't discoveries.

PRIORITY RULES (apply in order):

1. If the memory starts with "Fact:", "Concept:", "[X in #channel]", "Session:", or is biographical/descriptive about an entity → **Context**, not Insight.
2. If the memory describes a gotcha, failure mode, root cause, unexpected behavior, or something the author discovered through experience → **Insight**.
3. A security protocol, best practice, or recommendation is NOT a Decision unless someone explicitly chose it over an alternative → usually **Insight** or **Context**.
4. A tool description ("X does Y", "X is a library that...") is **Context**, not Insight, even if interesting.
5. A DM fragment or conversation excerpt is **Context**, not Decision.
6. A "how I set up X" or "how X works" explanation is **Context** unless it's framed as a reusable approach → then **Pattern**.
7. Only return **Preference** / **Style** / **Habit** when the memory is unambiguously one of those — these are narrow categories, don't stretch them.

Be conservative with Insight — it should mean genuine experiential learning, not any statement that sounds smart. If you're unsure between Insight and Context, pick Context.

Confidence: 0.95+ if the type is obvious, 0.75-0.9 if you had to apply a priority rule, 0.5-0.7 if genuinely ambiguous.

Return JSON with: {"type": "<type>", "confidence": <0.0-1.0>}"""

    def __init__(
        self,
        *,
        normalize_memory_type: Callable[[str], tuple[str, bool]],
        ensure_openai_client: Callable[[], None],
        get_openai_client: Callable[[], Any],
        classification_model: str,
        logger: Any,
        llm_first: bool = False,
        stats: Any = None,
    ) -> None:
        self._normalize_memory_type = normalize_memory_type
        self._ensure_openai_client = ensure_openai_client
        self._get_openai_client = get_openai_client
        self._classification_model = classification_model
        self._logger = logger
        self._llm_first = llm_first
        self._stats = stats

    def classify(self, content: str, *, use_llm: bool = True) -> tuple[str, float]:
        """Classify memory type and return confidence score.

        When ``llm_first`` is enabled, the LLM is consulted first and the regex
        patterns serve only as a fallback if the LLM is unavailable or fails.
        Default behavior keeps the legacy regex-first path.
        """
        if use_llm and self._llm_first:
            if self._stats is not None:
                self._stats.record_llm_attempt()
            try:
                result = self._classify_with_llm(content)
                if result:
                    if self._stats is not None:
                        self._stats.record_llm_success()
                    return result
            except Exception:
                self._logger.exception("LLM classification failed, falling back to regex")

        content_lower = content.lower()

        for memory_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    confidence = 0.6
                    matches = sum(1 for p in patterns if re.search(p, content_lower))
                    if matches > 1:
                        confidence = min(0.95, confidence + (matches * 0.1))
                    if self._stats is not None:
                        self._stats.record_pattern()
                    return memory_type, confidence

        if use_llm and not self._llm_first:
            llm_error: Optional[str] = None
            if self._stats is not None:
                self._stats.record_llm_attempt()
            try:
                result = self._classify_with_llm(content)
                if result:
                    if self._stats is not None:
                        self._stats.record_llm_success()
                    return result
                llm_error = "no usable LLM result (missing client, empty response, or invalid JSON)"
            except Exception as exc:
                self._logger.exception("LLM classification failed, using fallback")
                llm_error = str(exc)
            if self._stats is not None:
                self._stats.record_fallback(llm_error)

        return "Memory", 0.3

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Parse JSON from model output, tolerating ```json fences and prose.

        Some OpenAI-compatible providers (Gemini families on OpenRouter, certain
        Anthropic shims) ignore ``response_format=json_object`` and return prose
        with a JSON block embedded. Be lenient.
        """
        text = (text or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()
            if text.endswith("```"):
                text = text[:-3].strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise ValueError("No JSON object found in classification response")

    def _supports_json_response_format(self) -> bool:
        """Only request strict JSON mode for OpenAI-native models that honor it."""
        return self._classification_model.startswith(("gpt-", "o1-", "o3-", "o4-"))

    def _classify_with_llm(self, content: str) -> Optional[tuple[str, float]]:
        client = self._get_openai_client()
        if client is None:
            self._ensure_openai_client()
            client = self._get_openai_client()
        if client is None:
            return None

        try:
            extra_params: dict[str, Any] = {}
            uses_max_completion_tokens = self._classification_model.startswith(
                self.MAX_COMPLETION_PREFIXES
            )
            # 256 leaves headroom for reasoning models that "think" before the
            # JSON. Old budget of 50 left Gemini-class models with empty content.
            if uses_max_completion_tokens:
                extra_params["max_completion_tokens"] = 256
            else:
                extra_params["max_tokens"] = 256
                extra_params["temperature"] = 0.3

            request_kwargs: dict[str, Any] = {
                "model": self._classification_model,
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": content[:1000]},
                ],
                **extra_params,
            }
            if self._supports_json_response_format():
                request_kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**request_kwargs)

            raw_content = response.choices[0].message.content
            if not raw_content:
                self._logger.warning("LLM returned empty classification response")
                return None

            try:
                result = self._extract_json(raw_content)
            except (ValueError, json.JSONDecodeError, TypeError):
                self._logger.warning("LLM returned invalid JSON classification response")
                return None

            raw_type = result.get("type", "Memory")
            confidence = float(result.get("confidence", 0.7))

            memory_type, was_normalized = self._normalize_memory_type(raw_type)
            if not memory_type:
                self._logger.warning("LLM returned unmappable type '%s', using Context", raw_type)
                return "Context", 0.5

            if was_normalized and memory_type != raw_type:
                self._logger.debug("LLM type normalized '%s' -> '%s'", raw_type, memory_type)

            self._logger.info("LLM classified as %s (confidence: %.2f)", memory_type, confidence)
            return memory_type, confidence
        except Exception as exc:
            self._logger.warning("LLM classification failed: %s", exc)
            return None
