# claude_user_simulator.py
from __future__ import annotations

import os
import time
from typing import List, Optional

from anthropic import Anthropic

from .user_simulator import UserSimulator, STYLE_PERSONAS


class ClaudeStyleUserSimulator(UserSimulator):
    """
    Claude-backed style user simulator (online).

    Hardcoded to Claude Haiku 4.5 snapshot for consistency:
      - claude-haiku-4-5-20251001
    """

    def __init__(
        self,
        style: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        max_retries: int = 8,
        base_backoff_s: float = 0.75,
        api_key_env: str = "ANTHROPIC_API_KEY",
    ):
        if style not in STYLE_PERSONAS:
            raise ValueError(
                f"Unknown style '{style}'. Known styles: {list(STYLE_PERSONAS.keys())}"
            )

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is not set")

        self.client = Anthropic(api_key=api_key)

        # Hardcode Haiku 4.5 (snapshot). Use alias "claude-haiku-4-5" if you prefer auto-updates.
        self.model = "claude-haiku-4-5-20251001"

        self.style = style
        self.system_persona = STYLE_PERSONAS[style]

        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)

        self.max_retries = int(max_retries)
        self.base_backoff_s = float(base_backoff_s)

    def _one_call(self, raw_prompt: str, completion: str) -> str:
        system = (
            f"{self.system_persona}\n\n"
            "You are simulating the *user's next message* in a chat with an AI assistant.\n"
            "Rules:\n"
            "- Respond as the user with the USER PROFILE above in straightforward language.\n"
            "- Judge ONLY the assistant's response with respect to the preferences in the USER PROFILE; NOTHING ELSE\n"
            "- Do NOT answer the original request yourself.\n"
            "- Respond very briefly.\n"
            "- Do NOT give general feedback that does not relate to the USER PROFILE.\n"
            "- Output ONLY the user message text (no labels, no preface).\n"
            "- If the assistant answered well with respect to the USER PROFILE, you can say so briefly.\n"
        )

        user = (
            "Original user request:\n"
            f"{raw_prompt}\n\n"
            "Assistant response:\n"
            f"{completion}\n\n"
            "Write the user's next message:"
        )

        # NOTE: For Claude 4.5 models on some platforms, specifying both temperature and top_p may be invalid.
        # Keep it simple: temperature only.
        resp = self.client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        # Concatenate all text blocks
        parts: List[str] = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        text = "".join(parts).strip()

        # last-resort fallback to avoid empty strings
        return text if text else "Could you adjust the style to match what I prefer?"

    def generate_feedback(
        self,
        prompts: List[str],
        completions: List[str],
        rewards: Optional[List[float]] = None,
    ) -> List[str]:
        if len(prompts) != len(completions):
            raise ValueError("prompts and completions must have the same length")

        outs: List[str] = []
        for p, c in zip(prompts, completions):
            last_err: Optional[Exception] = None
            for attempt in range(self.max_retries):
                try:
                    outs.append(self._one_call(p, c))
                    break
                except Exception as e:
                    last_err = e
                    # simple exponential backoff
                    time.sleep(self.base_backoff_s * (2**attempt))
            else:
                raise RuntimeError(
                    f"Claude user simulation failed after {self.max_retries} retries: {last_err}"
                )
        return outs
