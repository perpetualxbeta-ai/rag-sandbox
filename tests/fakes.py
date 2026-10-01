"""Deterministic stand-ins for OpenAI so the pipeline can be tested offline.

* ``BagOfWordsEmbeddings`` - hashed bag-of-words vectors. Retrieval is lexical
  rather than semantic, so it is only good enough to exercise the plumbing and
  questions that share vocabulary with the handbook.
* ``RecordingLLM`` - a chat model that records every prompt it receives and
  answers with a fixed string, so tests can inspect what the chain sent.
* ``ExtractiveLLM`` - answers by echoing the retrieved context back. Used by
  ``evals/run_eval.py --offline`` to smoke-test the eval harness.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class BagOfWordsEmbeddings(Embeddings):
    dim: int = 1024

    def _vec(self, text: str) -> list[float]:
        text = unicodedata.normalize("NFKC", text).lower()
        v = [0.0] * self.dim
        for word in re.findall(r"[a-z0-9]+", text):
            if len(word) > 3 or word.isdigit():
                v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class RecordingLLM(BaseChatModel):
    reply: str = "FAKE ANSWER"
    calls: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "recording-fake"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kw: Any) -> ChatResult:
        self.calls.append(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])


class ExtractiveLLM(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "extractive-fake"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kw: Any) -> ChatResult:
        system = messages[0].content
        context = system.split("Handbook excerpts:", 1)[-1]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=context.strip()))])
