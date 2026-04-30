"""formosa_dual.data.tag_annotator — offline tag annotation pipeline.

Combines:
1. Aho-Corasick exact string matching over captions.
2. Metadata mapper (geo_tags, era_tags, article_title).
3. Optional LLM extractor for long-tail tags.

This module is used offline by ``scripts/annotate_tags.py``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from formosa_dual.data.tag_vocab import TagVocabulary
from formosa_dual.utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class LLMClient:
    """Minimal interface for LLM-based tag extraction.

    Subclass this and override :meth:`extract_tags` to plug in a real backend
    (HTTP, local Qwen, etc.).  Tests use a stub that returns a fixed list.
    """

    def extract_tags(self, caption: str, vocab_subset: list[str]) -> list[str]:
        """Extract cultural tags from *caption* using the LLM.

        Args:
            caption: Input caption text.
            vocab_subset: Candidate tags to consider (subset of vocabulary).

        Returns:
            List of extracted tag strings.
        """
        raise NotImplementedError(
            "LLMClient.extract_tags must be implemented by a concrete subclass."
        )


class TagAnnotator:
    """Combines Aho-Corasick + LLM extractor + metadata mapper.

    Used offline by ``scripts/annotate_tags.py``.

    Args:
        vocab: The tag vocabulary.
        use_aho_corasick: Whether to run Aho-Corasick exact-match.
        use_metadata: Whether to extract tags from record metadata fields.
        llm_client: Optional :class:`LLMClient` for LLM-based extraction.
        max_tags_per_image: Maximum number of tags returned per record.
    """

    def __init__(
        self,
        vocab: TagVocabulary,
        use_aho_corasick: bool = True,
        use_metadata: bool = True,
        llm_client: LLMClient | None = None,
        max_tags_per_image: int = 10,
    ) -> None:
        self._vocab = vocab
        self._use_aho_corasick = use_aho_corasick
        self._use_metadata = use_metadata
        self._llm_client = llm_client
        self._max_tags = max_tags_per_image

        self._automaton = None
        if use_aho_corasick:
            self._automaton = self._build_automaton(vocab)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_automaton(vocab: TagVocabulary):
        """Build an Aho-Corasick automaton from all vocabulary tags."""
        import ahocorasick

        A = ahocorasick.Automaton()
        for tag in vocab.tags:
            A.add_word(tag, tag)
        A.make_automaton()
        return A

    def _aho_corasick_extract(self, text: str) -> list[str]:
        """Run the automaton over *text* and return matched vocab tags."""
        if self._automaton is None or not text:
            return []
        found: list[str] = []
        for _, tag in self._automaton.iter(text):
            found.append(tag)
        return found

    def _metadata_extract(self, record: dict) -> list[str]:
        """Extract tags from record metadata fields (geo_tags, era_tags, title)."""
        tags: list[str] = []
        meta = record.get("metadata") or {}

        # Try article_title
        title = meta.get("article_title") or ""
        if title:
            tags.extend(self._aho_corasick_extract(title) if self._automaton else [])
            if title in self._vocab:
                tags.append(title)

        # geo_tags and era_tags
        for field in ("geo_tags", "era_tags"):
            for item in meta.get(field) or []:
                if item in self._vocab:
                    tags.append(item)

        # Also run AhoCorasick on ocr_text if present
        ocr = meta.get("ocr_text") or ""
        if ocr and self._automaton:
            tags.extend(self._aho_corasick_extract(ocr))

        return tags

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate(self, record: dict) -> list[str]:
        """Return the P_i tag list for *record*.

        Tags are de-duplicated, capped at *max_tags_per_image*, and all must
        be in the vocabulary.

        Args:
            record: A manifest record dict with at least ``caption`` and
                optionally ``metadata``.

        Returns:
            List of vocabulary tag strings.
        """
        seen: set[str] = set()
        tags: list[str] = []

        def _add(t: str) -> None:
            if t in self._vocab and t not in seen:
                seen.add(t)
                tags.append(t)

        # 1. Aho-Corasick on caption
        if self._use_aho_corasick:
            caption = record.get("caption") or ""
            for t in self._aho_corasick_extract(caption):
                _add(t)

        # 2. Metadata mapper
        if self._use_metadata:
            for t in self._metadata_extract(record):
                _add(t)

        # 3. LLM extractor
        if self._llm_client is not None and len(tags) < self._max_tags:
            caption = record.get("caption") or ""
            try:
                llm_tags = self._llm_client.extract_tags(caption, self._vocab.tags)
                for t in llm_tags:
                    _add(t)
            except Exception as exc:
                logger.error("LLM extraction failed for record %s: %s",
                             record.get("id", "<unknown>"), exc)
                raise

        return tags[: self._max_tags]
