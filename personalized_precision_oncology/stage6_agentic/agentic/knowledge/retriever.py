"""
Deterministic Lexical Evidence Retriever for Stage 6 Agentic AI.

Provides BM25/TF-IDF style lexical retrieval augmented with oncology ontology expansion
(from oncology_dictionary.py) to reliably match queries such as:
- "EGFR L858R NSCLC"
- "high toxicity cisplatin"
- "progression with resistance"
- "PD-L1 treatment context"

Zero LLM dependencies. 100% deterministic, thread-safe, and offline.
Returns EVIDENCE_NOT_FOUND when no supporting evidence exists.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Set
from personalized_precision_oncology.stage6_agentic.agentic.knowledge.evidence_store import EvidenceStore
from personalized_precision_oncology.stage6_agentic.agentic.knowledge.oncology_dictionary import OncologyDictionary
from personalized_precision_oncology.stage6_agentic.agentic.knowledge.schemas import (
    EvidenceLevel,
    EvidenceRecord,
    RetrievalResponse,
    RetrievalResult,
)


class KnowledgeRetriever:
    """
    Deterministic lexical retrieval engine over the Stage 6 EvidenceStore.
    """

    def __init__(
        self,
        store: Optional[EvidenceStore] = None,
        dictionary: Optional[OncologyDictionary] = None
    ) -> None:
        self._store = store if store is not None else EvidenceStore(populate_knowledge_base=True)
        self._dictionary = dictionary if dictionary is not None else OncologyDictionary()

    @property
    def store(self) -> EvidenceStore:
        return self._store

    @property
    def dictionary(self) -> OncologyDictionary:
        return self._dictionary

    def _tokenize(self, text: str) -> List[str]:
        """Normalize and tokenize text into lowercase word tokens."""
        clean = re.sub(r"[^\w\s\.\-]", " ", text.lower())
        tokens = [t.strip() for t in clean.split() if len(t.strip()) > 1]
        return tokens

    def _compute_idf(self, corpus_tokens: List[List[str]], vocabulary: Set[str]) -> Dict[str, float]:
        """Compute standard BM25 inverse document frequency for terms in vocabulary."""
        n_docs = len(corpus_tokens)
        idf: Dict[str, float] = {}
        for term in vocabulary:
            doc_count = sum(1 for doc in corpus_tokens if term in doc)
            # Standard smooth BM25 IDF formulation
            idf[term] = math.log((n_docs - doc_count + 0.5) / (doc_count + 0.5) + 1.0)
        return idf

    def _get_evidence_level_multiplier(self, level: EvidenceLevel) -> float:
        """Weight evidence quality in score calculation."""
        multipliers = {
            EvidenceLevel.LEVEL_A: 1.25,
            EvidenceLevel.LEVEL_B: 1.15,
            EvidenceLevel.EXPERT_CONSENSUS: 1.10,
            EvidenceLevel.MODEL_INFERENCE: 1.05,
            EvidenceLevel.LEVEL_C: 1.0,
            EvidenceLevel.LEVEL_D: 0.95,
        }
        return multipliers.get(level, 1.0)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_relevance: float = 0.05,
        stage_filter: Optional[str] = None,
        category_filter: Optional[str] = None
    ) -> RetrievalResponse:
        """
        Execute deterministic lexical retrieval for a query.

        Args:
            query: Input clinical query string.
            top_k: Maximum number of ranked results to return.
            min_relevance: Relevance score threshold below which results are excluded.
            stage_filter: Optional filter by EvidenceStage string.
            category_filter: Optional filter by EvidenceCategory string.

        Returns:
            RetrievalResponse with ranked results or status='EVIDENCE_NOT_FOUND'.
        """
        records = self._store.get_all_records()
        if not records or not query.strip():
            return RetrievalResponse(
                query=query,
                status="EVIDENCE_NOT_FOUND",
                total_results=0,
                results=[]
            )

        # 1. Expand query terms using ontology dictionary
        expanded_query_terms = self._dictionary.expand_query_terms(query)
        query_tokens = self._tokenize(query)
        all_query_terms = set(query_tokens).union(expanded_query_terms)

        # 2. Filter records if requested
        candidate_records: List[EvidenceRecord] = []
        for r in records:
            if stage_filter and r.source_stage.value != stage_filter:
                continue
            if category_filter and r.category.lower() != category_filter.lower():
                continue
            candidate_records.append(r)

        if not candidate_records:
            return RetrievalResponse(
                query=query,
                status="EVIDENCE_NOT_FOUND",
                total_results=0,
                results=[]
            )

        # 3. Build token representations for BM25 computation
        doc_tokens_list: List[List[str]] = []
        doc_topic_tokens_list: List[List[str]] = []
        doc_matched_tokens_list: List[List[str]] = []

        for r in candidate_records:
            text_toks = self._tokenize(r.evidence_text)
            topic_toks = self._tokenize(r.topic)
            matched_toks = []
            for m in r.matched_terms:
                matched_toks.extend(self._tokenize(m))

            doc_tokens_list.append(text_toks)
            doc_topic_tokens_list.append(topic_toks)
            doc_matched_tokens_list.append(matched_toks)

        # Average doc length for BM25
        avg_doc_len = sum(len(d) for d in doc_tokens_list) / max(len(doc_tokens_list), 1)
        k1 = 1.2
        b = 0.75

        # Compute IDF over candidate corpus
        combined_corpus = [
            doc_tokens_list[i] + doc_topic_tokens_list[i] + doc_matched_tokens_list[i]
            for i in range(len(candidate_records))
        ]
        idf_dict = self._compute_idf(combined_corpus, all_query_terms)

        # 4. Score each document
        scored_results: List[RetrievalResult] = []

        for i, rec in enumerate(candidate_records):
            text_tokens = doc_tokens_list[i]
            topic_tokens = doc_topic_tokens_list[i]
            matched_terms_tokens = doc_matched_tokens_list[i]
            doc_len = len(text_tokens)

            tf_text = Counter(text_tokens)
            tf_topic = Counter(topic_tokens)
            tf_matched = Counter(matched_terms_tokens)

            score = 0.0
            matched_terms_in_doc: List[str] = []

            for term in all_query_terms:
                idf = idf_dict.get(term, 0.0)
                if idf <= 0.0:
                    continue

                f_text = tf_text.get(term, 0)
                f_topic = tf_topic.get(term, 0)
                f_matched = tf_matched.get(term, 0)

                # Track matched terms
                if f_text > 0 or f_topic > 0 or f_matched > 0:
                    matched_terms_in_doc.append(term)

                # BM25 core term score for text body
                if f_text > 0:
                    bm25_text = idf * (f_text * (k1 + 1)) / (f_text + k1 * (1 - b + b * (doc_len / avg_doc_len)))
                    score += bm25_text

                # Boost for matches in Topic (title) and Matched Terms
                if f_topic > 0:
                    score += 3.0 * idf * f_topic
                if f_matched > 0:
                    score += 2.0 * idf * f_matched

            # Check for exact multi-word query substrings in text or topic
            query_lower = query.strip().lower()
            if query_lower in rec.topic.lower():
                score += 5.0
            elif query_lower in rec.evidence_text.lower():
                score += 3.0

            # Evidence level multiplier
            multiplier = self._get_evidence_level_multiplier(rec.evidence_level)
            score *= multiplier

            if score >= min_relevance and matched_terms_in_doc:
                scored_results.append(
                    RetrievalResult(
                        evidence_id=rec.evidence_id,
                        relevance_score=round(score, 4),
                        matched_terms=sorted(list(set(matched_terms_in_doc))),
                        source=rec.source_stage.value,
                        evidence_level=rec.evidence_level.value,
                        safety_note=rec.safety_note,
                        provenance=rec.provenance,
                        record=rec
                    )
                )

        # 5. Deterministic sorting: by relevance_score descending, then evidence_id ascending
        scored_results.sort(key=lambda x: (-x.relevance_score, x.evidence_id))
        top_results = scored_results[:top_k]

        if not top_results:
            return RetrievalResponse(
                query=query,
                status="EVIDENCE_NOT_FOUND",
                total_results=0,
                results=[]
            )

        return RetrievalResponse(
            query=query,
            status="SUCCESS",
            total_results=len(top_results),
            results=top_results
        )
