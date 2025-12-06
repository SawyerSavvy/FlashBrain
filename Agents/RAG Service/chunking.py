"""
Hybrid Semantic Document Chunking

Implements a three-stage chunking approach:
1. Structure Parsing - Parse document hierarchy (headers, sections)
2. Embedding-Based Refinement - Detect semantic boundaries using embeddings
3. Optional LLM Validation - Validate boundaries for critical documents

Based on: Chunking Approach.md
"""

import re
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class HybridChunker:
    """
    Hybrid semantic chunking combining structure parsing,
    embedding-based refinement, and optional LLM validation.
    """

    def __init__(
        self,
        embeddings_model,
        llm=None,
        max_chunk_size: int = 1000,
        min_chunk_tokens: int = 512,
        embedding_threshold: float = 0.5,
        validate_with_llm: bool = False
    ):
        """
        Initialize the hybrid chunker.

        Args:
            embeddings_model: Embedding model for semantic similarity
            llm: Optional LLM for validation (only for critical docs)
            max_chunk_size: Maximum chunk size in characters
            min_chunk_tokens: Target minimum chunk size in tokens (approx.)
            embedding_threshold: Similarity threshold for splitting (0-1)
            validate_with_llm: Whether to validate with LLM by default
        """
        self.embeddings_model = embeddings_model
        self.llm = llm
        self.max_chunk_size = max_chunk_size
        self.min_chunk_tokens = min_chunk_tokens
        self.embedding_threshold = embedding_threshold
        self.validate_with_llm = validate_with_llm

    async def chunk_document(
        self,
        text: str,
        doc_type: str = "auto",
        format_type: str = "auto",
        metadata: Optional[Dict[str, Any]] = None,
        force_llm_validation: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Complete hybrid chunking pipeline.

        Args:
            text: Document text
            doc_type: Document type ('technical', 'legal', 'security', etc.)
            format_type: Format ('markdown', 'html', 'pdf', 'plain')
            metadata: Document metadata
            force_llm_validation: Force LLM validation regardless of doc_type

        Returns:
            List of enriched chunks with metadata
        """
        metadata = metadata or {}

        # Auto-detect format if needed
        if format_type == "auto":
            format_type = self._detect_format(text)

        # STAGE 1: Structure Parsing
        logger.info(f"Stage 1: Parsing {format_type} structure")
        initial_chunks = await self.parse_document_structure(text, format_type)
        logger.info(f"Created {len(initial_chunks)} initial chunks from structure")

        # STAGE 2: Embedding-Based Refinement
        logger.info("Stage 2: Refining with embedding similarity")
        refined_chunks = await self.refine_chunks_with_embeddings(
            initial_chunks,
            threshold=self.embedding_threshold,
            max_chunk_size=self.max_chunk_size
        )
        logger.info(f"Refined to {len(refined_chunks)} chunks")

        # STAGE 3: Optional LLM Validation
        should_validate = (
            force_llm_validation or
            self.validate_with_llm or
            doc_type in ["legal", "security", "compliance"]
        )

        if should_validate and self.llm:
            logger.info("Stage 3: Validating boundaries with LLM")
            final_chunks = await self.validate_boundaries_with_llm(
                refined_chunks,
                doc_type
            )
            logger.info(f"Validated {len(final_chunks)} chunks")
        else:
            final_chunks = refined_chunks

        # STAGE 4: Enrich with metadata
        enriched_chunks = self._enrich_chunks_with_metadata(
            final_chunks,
            doc_type,
            format_type,
            metadata
        )

        return enriched_chunks

    def _detect_format(self, text: str) -> str:
        """Auto-detect document format."""
        # Check for HTML tags
        if bool(BeautifulSoup(text, "html.parser").find()):
            return "html"
        # Check for markdown headers
        if re.search(r'^#{1,6}\s+.+$', text, re.MULTILINE):
            return "markdown"
        # Default to plain text
        return "plain"

    # ================================================================
    # STAGE 1: STRUCTURE PARSING
    # ================================================================

    async def parse_document_structure(
        self,
        text: str,
        format_type: str
    ) -> List[Dict[str, Any]]:
        """
        Parse document structure based on format type.

        Args:
            text: Document text
            format_type: Format ('markdown', 'html', 'pdf', 'plain')

        Returns:
            List of initial chunks with hierarchy information
        """
        if format_type == "markdown":
            return await self.parse_markdown_headers(text)
        elif format_type == "html":
            return await self.parse_html_tags(text)
        elif format_type == "pdf":
            return await self.parse_pdf_sections(text)
        else:
            # Plain text: create fixed-size chunks
            return await self.fixed_size_chunks(text, size=self.max_chunk_size)

    async def parse_markdown_headers(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse markdown document by headers.

        Creates chunks that respect markdown hierarchy (# > ## > ###).
        """
        chunks = []
        current_chunk = {
            "content": [],
            "headers": [],
            "level": None,
            "hierarchy": []
        }

        lines = text.split('\n')

        for line in lines:
            # Check for header
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)

            if header_match:
                level = len(header_match.group(1))  # Number of # symbols
                title = header_match.group(2).strip()

                # If going back up hierarchy, save current chunk
                if current_chunk["level"] is not None and level <= current_chunk["level"]:
                    if current_chunk["content"]:
                        chunks.append(self._finalize_chunk(current_chunk))

                    # Start new chunk with updated hierarchy
                    current_chunk = {
                        "content": [line],
                        "headers": [title],
                        "level": level,
                        "hierarchy": self._build_hierarchy(chunks, level, title)
                    }
                else:
                    # Going deeper in hierarchy
                    current_chunk["headers"].append(title)
                    current_chunk["level"] = level
                    current_chunk["content"].append(line)
                    current_chunk["hierarchy"] = self._build_hierarchy(chunks, level, title)

            else:
                # Regular content line
                current_chunk["content"].append(line)

            # Split if chunk exceeds max size
            content_length = len('\n'.join(current_chunk["content"]))
            if content_length > self.max_chunk_size:
                chunks.append(self._finalize_chunk(current_chunk))
                current_chunk = {
                    "content": [],
                    "headers": current_chunk["headers"].copy(),  # Preserve headers
                    "level": current_chunk["level"],
                    "hierarchy": current_chunk["hierarchy"].copy()
                }

        # Add final chunk
        if current_chunk["content"]:
            chunks.append(self._finalize_chunk(current_chunk))

        return chunks

    async def parse_html_tags(self, html: str) -> List[Dict[str, Any]]:
        """
        Parse HTML document by semantic tags.

        Extracts sections, articles, and divs with headers.
        """
        soup = BeautifulSoup(html, 'html.parser')
        chunks = []

        # Find semantic sections
        sections = soup.find_all(['section', 'article', 'div'])

        for section in sections:
            # Skip if no meaningful content
            text = section.get_text(strip=True)
            if len(text) < 50:  # Skip very short sections
                continue

            # Extract title from headers
            header = section.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            title = header.get_text(strip=True) if header else "Untitled Section"

            # Get header level
            level = int(header.name[1]) if header else 0

            chunks.append({
                "content": text,
                "headers": [title],
                "level": level,
                "hierarchy": [title],
                "tag": section.name
            })

        return chunks

    async def parse_pdf_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse PDF document by detected sections.

        Uses heuristics to detect section boundaries in plain PDF text.
        """
        # PDF often has implicit structure
        # Look for patterns like:
        # - ALL CAPS LINES (section headers)
        # - Numbered sections (1. Introduction, 2. Methods)
        # - Bold indicators (if preserved in text extraction)

        chunks = []
        current_chunk = {"content": [], "headers": [], "level": None, "hierarchy": []}

        lines = text.split('\n')

        for line in lines:
            line_stripped = line.strip()

            # Detect section headers (heuristics)
            is_header = False
            header_title = None
            level = None

            # Pattern 1: ALL CAPS (likely a header)
            if line_stripped.isupper() and len(line_stripped.split()) > 1:
                is_header = True
                header_title = line_stripped.title()
                level = 1

            # Pattern 2: Numbered sections (1. Introduction)
            # Only treat as header if there are multiple levels (e.g., 1.1, 2.3.1), to avoid splitting simple numbered lists
            numbered_match = re.match(r'^(\d+\.\d+(?:\.\d+)*)\s+(.+)$', line_stripped)
            if numbered_match:
                is_header = True
                header_title = numbered_match.group(2)
                level = numbered_match.group(1).count('.')

            if is_header and header_title:
                # Save previous chunk
                if current_chunk["content"]:
                    chunks.append(self._finalize_chunk(current_chunk))

                # Start new chunk
                current_chunk = {
                    "content": [line],
                    "headers": [header_title],
                    "level": level,
                    "hierarchy": [header_title]
                }
            else:
                current_chunk["content"].append(line)

            # Split if too large
            if len('\n'.join(current_chunk["content"])) > self.max_chunk_size:
                chunks.append(self._finalize_chunk(current_chunk))
                current_chunk = {
                    "content": [],
                    "headers": current_chunk["headers"].copy(),
                    "level": current_chunk["level"],
                    "hierarchy": current_chunk["hierarchy"].copy()
                }

        # Add final chunk
        if current_chunk["content"]:
            chunks.append(self._finalize_chunk(current_chunk))

        return chunks if chunks else await self.fixed_size_chunks(text, size=self.max_chunk_size)

    async def fixed_size_chunks(
        self,
        text: str,
        size: int = 1000,
        overlap: int = None
    ) -> List[Dict[str, Any]]:
        """
        Fallback: Create fixed-size chunks for plain text.

        Used when no structure is detected.
        """
        # Aim for slight overlap (~10% of chunk size), defaulting to token-aligned window behavior
        effective_overlap = overlap if overlap is not None else max(0, size // 10)

        chunks = []
        words = text.split()
        start = 0
        n = len(words)

        while start < n:
            current_chunk = []
            current_length = 0
            i = start

            while i < n:
                word_length = len(words[i]) + (1 if current_chunk else 0)  # space before word if not first
                if current_length + word_length > size and current_chunk:
                    break
                current_chunk.append(words[i])
                current_length += word_length
                i += 1

            if not current_chunk:
                # Handle extremely long single word
                current_chunk.append(words[start])
                i = start + 1

            chunks.append({
                "content": ' '.join(current_chunk),
                "headers": [],
                "level": None,
                "hierarchy": []
            })

            if i >= n:
                break

            # Compute how many words to overlap based on character budget
            if effective_overlap > 0:
                overlap_chars = 0
                backtrack_words = 0
                j = len(current_chunk) - 1
                while j >= 0 and overlap_chars < effective_overlap:
                    overlap_chars += len(current_chunk[j]) + 1  # include space
                    backtrack_words += 1
                    j -= 1
                start = max(0, i - backtrack_words)
            else:
                start = i

            if start >= n:
                break

        return chunks

    def _finalize_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Convert chunk content list to string."""
        return {
            **chunk,
            "content": '\n'.join(chunk["content"]).strip()
        }

    def _build_hierarchy(
        self,
        existing_chunks: List[Dict[str, Any]],
        level: int,
        title: str
    ) -> List[str]:
        """Build breadcrumb hierarchy for a chunk."""
        # Simple hierarchy: just track current level
        hierarchy = [title]
        return hierarchy

    # ================================================================
    # STAGE 2: EMBEDDING-BASED REFINEMENT
    # ================================================================

    async def refine_chunks_with_embeddings(
        self,
        chunks: List[Dict[str, Any]],
        threshold: float = 0.5,
        max_chunk_size: int = 1000,
        min_chunk_tokens: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Refine chunks using embedding similarity.

        For large chunks, detect topic shifts by embedding sentences
        and splitting where similarity drops below threshold.
        """
        min_tokens = min_chunk_tokens or self.min_chunk_tokens
        refined_chunks = []

        for chunk in chunks:
            content = chunk["content"]
            token_len = self._estimate_tokens(content)

            # If chunk is within the target window, keep as-is (merge later if tiny)
            if token_len <= min_tokens:
                refined_chunks.append(chunk)
                continue

            # Large chunk: detect sub-boundaries using paragraphs
            sub_chunks = await self.split_by_embedding_breakpoints(
                content,
                threshold,
                max_chunk_size,
                min_tokens
            )

            for sub_content in sub_chunks:
                refined_chunks.append({
                    **chunk,
                    "content": sub_content
                })

        # Merge adjacent chunks to reach at least min_tokens and maintain slight overlap
        merged = self._merge_to_min_tokens(refined_chunks, min_tokens, max_chunk_size)
        return merged

    async def split_by_embedding_breakpoints(
        self,
        text: str,
        threshold: float,
        max_chunk_size: int,
        min_chunk_tokens: int
    ) -> List[str]:
        """
        Split text by detecting semantic boundaries using embeddings.

        Returns list of sub-chunk texts.
        """
        # Step 1: Split into paragraphs to preserve higher-level context
        paragraphs = self._split_into_paragraphs(text)

        if len(paragraphs) <= 1:
            return [text]

        # Step 2: Embed each paragraph
        try:
            embeddings = await self.embeddings_model.aembed_documents(paragraphs)
        except Exception as e:
            logger.warning(f"Failed to embed sentences: {e}. Falling back to original chunk.")
            return [text]

        # Step 3: Calculate cosine similarity between consecutive paragraphs
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(
                [embeddings[i]],
                [embeddings[i + 1]]
            )[0][0]
            similarities.append(sim)

        # Step 4: Identify breakpoints (low similarity = topic change)
        breakpoints = [0]
        for i, sim in enumerate(similarities):
            if sim < threshold:
                breakpoints.append(i + 1)
        breakpoints.append(len(paragraphs))

        # Step 5: Create sub-chunks at breakpoints
        sub_chunks = []
        for i in range(len(breakpoints) - 1):
            start_idx = breakpoints[i]
            end_idx = breakpoints[i + 1]
            chunk_text = '\n\n'.join(paragraphs[start_idx:end_idx])
            sub_chunks.append(chunk_text)

        # Step 6: Merge tiny chunks to avoid fragmentation while respecting limits
        merged_chunks = self._merge_small_chunks(sub_chunks, min_chunk_tokens, max_chunk_size)

        return merged_chunks

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate using whitespace split."""
        return len(text.split())

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs separated by blank lines."""
        return [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple heuristics."""
        # Simple sentence splitting (can be improved with nltk/spacy)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _merge_small_chunks(
        self,
        chunks: List[str],
        min_tokens: int,
        max_size: Optional[int]
    ) -> List[str]:
        """Merge chunks that are too small to avoid fragmentation."""
        merged = []
        current_chunk = ""

        for chunk in chunks:
            prospective = chunk if not current_chunk else current_chunk + "\n\n" + chunk
            token_len = self._estimate_tokens(prospective)
            if token_len < min_tokens or (max_size and len(prospective) <= max_size):
                current_chunk = prospective
            else:
                if current_chunk:
                    merged.append(current_chunk)
                current_chunk = chunk

        # Add final chunk
        if current_chunk:
            merged.append(current_chunk)

        return merged

    def _merge_to_min_tokens(
        self,
        chunks: List[Dict[str, Any]],
        min_tokens: int,
        max_size: Optional[int]
    ) -> List[Dict[str, Any]]:
        """
        Merge adjacent chunks to enforce minimum token length,
        preserving order and chunk metadata from the first chunk in the merge.
        """
        if not chunks:
            return []

        merged: List[Dict[str, Any]] = []
        current = chunks[0]

        for next_chunk in chunks[1:]:
            combined_content = f"{current['content']}\n\n{next_chunk['content']}"
            combined_tokens = self._estimate_tokens(combined_content)

            if self._estimate_tokens(current["content"]) < min_tokens:
                # Merge to reach minimum
                current = {
                    **current,
                    "content": combined_content
                }
            else:
                # Allow slight overlap by starting next chunk from previous tail if needed
                merged.append(current)
                current = next_chunk

        merged.append(current)
        return merged

    # ================================================================
    # STAGE 3: OPTIONAL LLM VALIDATION
    # ================================================================

    async def validate_boundaries_with_llm(
        self,
        chunks: List[Dict[str, Any]],
        doc_type: str
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to validate chunk boundaries.

        Only applies to critical document types.
        """
        if not self.llm:
            logger.warning("LLM validation requested but no LLM provided")
            return chunks

        if doc_type not in ["legal", "security", "compliance"]:
            # Skip for non-critical docs
            return chunks

        validated_chunks = []

        for chunk in chunks:
            # Ask LLM to validate boundary
            prompt = f"""Review this document chunk. Does it form a complete semantic unit?
Should it be split, merged, or adjusted for better semantic coherence?

Current chunk:
{chunk['content']}

Respond with JSON:
{{
  "is_valid": true/false,
  "suggestion": "explanation of why valid/invalid",
  "action": "keep" | "split" | "merge"
}}"""

            try:
                response = await self.llm.ainvoke(prompt)
                content = response.content.strip()

                # Extract JSON
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    import json
                    feedback = json.loads(json_match.group())

                    if feedback.get("is_valid", True):
                        # Chunk is valid
                        validated_chunks.append(chunk)
                    else:
                        # LLM suggests adjustment
                        # For now, keep original (can implement split/merge logic)
                        logger.info(f"LLM suggests adjustment: {feedback.get('suggestion')}")
                        validated_chunks.append(chunk)
                else:
                    # Couldn't parse response, keep chunk
                    validated_chunks.append(chunk)

            except Exception as e:
                logger.warning(f"LLM validation failed: {e}. Keeping chunk as-is.")
                validated_chunks.append(chunk)

        return validated_chunks

    # ================================================================
    # STAGE 4: METADATA ENRICHMENT
    # ================================================================

    def _enrich_chunks_with_metadata(
        self,
        chunks: List[Dict[str, Any]],
        doc_type: str,
        format_type: str,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Add breadcrumbs and metadata to chunks."""
        enriched = []

        for i, chunk in enumerate(chunks):
            enriched.append({
                "chunk_index": i,
                "content": chunk["content"],
                "breadcrumb": self._build_breadcrumb(chunk.get("headers", [])),
                "section": chunk.get("headers", ["Root"])[-1] if chunk.get("headers") else "Root",
                "metadata": {
                    **metadata,
                    "doc_type": doc_type,
                    "format": format_type,
                    "headers": chunk.get("headers", []),
                    "hierarchy": chunk.get("hierarchy", []),
                    "level": chunk.get("level")
                }
            })

        return enriched

    def _build_breadcrumb(self, headers: List[str]) -> str:
        """Build breadcrumb path from headers."""
        return " > ".join(headers) if headers else "Root"


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def calculate_similarity_matrix(embeddings: List[List[float]]) -> np.ndarray:
    """
    Calculate pairwise cosine similarity matrix for embeddings.

    Args:
        embeddings: List of embedding vectors

    Returns:
        Similarity matrix (n x n)
    """
    embeddings_array = np.array(embeddings)
    return cosine_similarity(embeddings_array)


def detect_topic_boundaries(
    similarities: List[float],
    threshold: float = 0.5
) -> List[int]:
    """
    Detect topic boundaries based on similarity drops.

    Args:
        similarities: List of consecutive sentence similarities
        threshold: Similarity threshold for boundary

    Returns:
        List of boundary indices
    """
    boundaries = [0]
    for i, sim in enumerate(similarities):
        if sim < threshold:
            boundaries.append(i + 1)
    return boundaries
