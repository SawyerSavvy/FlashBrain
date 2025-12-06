# Hybrid Semantic Document Chunking Summary

A production-ready approach that combines three techniques for intelligent document splitting:

## Three-Step Process

**Step 1: Structure Parsing**
- Parse document format (Markdown headers, HTML tags, PDF sections)
- Respect natural hierarchy (H1 → H2 → H3)
- Extract section titles and breadcrumbs
- Creates initial coarse-grained chunks that preserve organization

**Step 2: Embedding-Based Refinement**
- Aim for ~512-token chunks with slight overlap; smaller pieces are merged with neighbors
- For larger chunks, use embedding similarity over paragraphs to detect topic shifts
- Calculate cosine similarity between consecutive paragraphs
- Split where similarity drops below threshold (e.g., 0.5)
- Detects semantic boundaries within sections while keeping chunks substantial enough for context

**Step 3: Optional LLM Validation** *(for critical documents)*
- For legal, security, or high-stakes docs, use LLM to verify/correct boundaries
- Ensures accuracy on complex, ambiguous content
- Only when cost/accuracy tradeoff justifies it

***

## Key Benefits

| Aspect | Benefit |
|--------|---------|
| **Accuracy** | Respects both document structure and semantic meaning |
| **Cost** | Efficient (structure + embeddings; LLM only when needed) |
| **Speed** | Parallel embedding computation; fast in production |
| **Context Preservation** | Maintains hierarchy breadcrumbs for better retrieval |
| **Scalability** | Works on any document type (Markdown, HTML, PDF, plain text) |

***

## Output

Each chunk contains:
- **Content:** The actual text
- **Breadcrumb:** `Chapter 1 > Section 2.3 > Subsection` (hierarchy path)
- **Section metadata:** Section title, document type, source
- **Headers:** All parent header levels

This metadata gets stored alongside embeddings in your vector DB, enabling filtered retrieval and better context injection into RAG prompts.

***

## When to Use Each Technique

| Technique | Use Case |
|-----------|----------|
| Structure parsing alone | Well-formatted docs (technical docs, blogs, knowledge bases) |
| + Embedding refinement | Mixed-quality documents, unclear structure |
| + LLM validation | Legal contracts, security policies, regulatory docs |

**Recommendation:** Start with Structure + Embeddings for 80% of documents; add LLM validation only for critical 20%.

# Hybrid Semantic Document Chunking: Full Summary with Pseudocode

## Overview

A three-stage intelligent document splitting system that combines **structural awareness**, **semantic detection**, and **optional validation** to create meaningful, context-preserving chunks for RAG systems.

***

## Stage 1: Structure Parsing

**Description:**
Parse the document's natural hierarchy (headers, sections, HTML tags) to create initial chunks that respect the document's organization. This preserves context and document flow.

**Pseudocode:**

```
function parseDocumentStructure(document, format):
    if format == "markdown":
        return parseMarkdownHeaders(document)
    else if format == "html":
        return parseHTMLTags(document)
    else if format == "pdf":
        return parsePDFSections(document)
    else:
        return fixedSizeChunks(document, size=1000)

function parseMarkdownHeaders(text):
    chunks = []
    currentChunk = {content: [], headers: [], level: null}
    
    for each line in text.split('\n'):
        headerMatch = regex match "^(#+) (.+)$"
        
        if headerMatch found:
            level = length of header marker
            title = header text
            
            // If we're going back up the hierarchy, save current chunk
            if currentChunk.level != null AND level <= currentChunk.level:
                chunks.append(currentChunk)
                currentChunk = {content: [], headers: [title], level: level}
            else:
                currentChunk.headers.append(title)
                currentChunk.level = level
        
        currentChunk.content.append(line)
        
        // Split if chunk exceeds max size
        if length of currentChunk.content > 1000:
            chunks.append(currentChunk)
            currentChunk = {content: [], headers: [], level: null}
    
    chunks.append(currentChunk)  // Add final chunk
    return chunks

function parseHTMLTags(html):
    dom = parse(html)
    chunks = []
    
    for each section in dom.findAll(['section', 'article', 'div.content']):
        text = section.extractText()
        title = section.findFirst(['h1', 'h2', 'h3']).getText()
        
        chunks.append({
            content: text,
            title: title,
            tag: section.tagName
        })
    
    return chunks
```

**Output:** Initial chunks with preserved hierarchy and breadcrumb information.

***

## Stage 2: Embedding-Based Refinement

**Description:**
Aim for ~512-token windows with slight overlap. Large chunks are refined semantically at the paragraph level; small chunks are merged to hit the target size.

**Pseudocode:**

```
function refineChunksWithEmbeddings(chunks, threshold=0.5, maxChunkSize=1000, minTokens=512):
    refinedChunks = []
    embedder = loadEmbeddingModel("text-embedding-3-small")
    
    for each chunk in chunks:
        if tokenLength(chunk.content) > minTokens:
            // Large chunk: detect sub-topic boundaries using paragraphs
            subChunks = splitByEmbeddingBreakpoints(chunk, embedder, threshold, minTokens)
            for each subChunk in subChunks:
                refinedChunks.append({...chunk, content: subChunk})
        else:
            // Small chunk: keep as-is (will merge later)
            refinedChunks.append(chunk)
    
    return mergeToMinTokens(refinedChunks, minTokens)

function splitByEmbeddingBreakpoints(chunk, embedder, threshold, minTokens):
    text = chunk.content
    paragraphs = splitOnBlankLines(text)
    subChunks = []
    
    // Step 1: Embed each paragraph
    embeddings = []
    for each paragraph in paragraphs:
        embedding = embedder.embed(paragraph)
        embeddings.append(embedding)
    
    // Step 2: Calculate similarity between consecutive paragraphs
    similarities = []
    for i = 0 to length(embeddings) - 2:
        sim = cosineSimilarity(embeddings[i], embeddings[i+1])
        similarities.append(sim)
    
    // Step 3: Identify breakpoints (low similarity = topic change)
    breakpoints = [0]
    for i, sim in similarities:
        if sim < threshold:
            breakpoints.append(i + 1)
    breakpoints.append(length(paragraphs))
    
    // Step 4: Create chunks at breakpoints
    for i = 0 to length(breakpoints) - 2:
        startIdx = breakpoints[i]
        endIdx = breakpoints[i + 1]
        chunkText = join(paragraphs[startIdx:endIdx], "\n\n")
        subChunks.append(chunkText)
    
    // Step 5: Merge tiny chunks to avoid fragmentation and enforce minTokens
    return mergeToMinTokens(subChunks, minTokens)
```

**Output:** Refined chunks with semantic boundaries detected within sections.

***

## Stage 3: Optional LLM Validation

**Description:**
For critical documents (legal, security, compliance), use an LLM to verify chunk boundaries are correct and preserve meaning. Only applied when accuracy is critical.

**Pseudocode:**

```
function validateBoundariesWithLLM(chunks, llm, docType):
    if docType NOT in ["legal", "security", "compliance"]:
        return chunks  // Skip for non-critical docs
    
    validatedChunks = []
    
    for each chunk in chunks:
        // Ask LLM to validate and potentially adjust boundary
        prompt = """
        Review this chunk. Does it form a complete semantic unit?
        Should it be split, merged, or adjusted?
        
        Current chunk:
        {chunk.content}
        
        Return JSON: {is_valid: bool, suggestion: string}
        """
        
        response = llm.invoke(prompt)
        feedback = parseJSON(response)
        
        if feedback.is_valid:
            validatedChunks.append(chunk)
        else:
            // Re-split based on LLM suggestion
            adjustedChunk = applyLLMSuggestion(chunk, feedback)
            validatedChunks.append(adjustedChunk)
    
    return validatedChunks
```

**Output:** Validated chunks with LLM-approved boundaries.

***

## Complete Hybrid Pipeline

**Pseudocode:**

```
function hybridSemanticChunking(document, docType="auto", config={}):
    // Configuration defaults
    maxChunkSize = config.maxChunkSize || 1000
    embeddingThreshold = config.embeddingThreshold || 0.5
    validateWithLLM = config.validateWithLLM || false
    
    // STAGE 1: Parse structure
    detectedFormat = detectFormat(document)
    initialChunks = parseDocumentStructure(document, detectedFormat)
    
    // STAGE 2: Refine with embeddings
    refinedChunks = refineChunksWithEmbeddings(
        initialChunks,
        threshold=embeddingThreshold,
        maxChunkSize=maxChunkSize
    )
    
    // STAGE 3: Optionally validate with LLM
    if validateWithLLM OR docType in ["legal", "security"]:
        llm = loadLLM("gpt-4-turbo")
        finalChunks = validateBoundariesWithLLM(refinedChunks, llm, docType)
    else:
        finalChunks = refinedChunks
    
    // STAGE 4: Enrich with metadata
    enrichedChunks = []
    for each chunk in finalChunks:
        enrichedChunk = {
            content: chunk.content,
            breadcrumb: buildBreadcrumb(chunk.headers),
            section: chunk.headers[-1] if chunk.headers else "Root",
            metadata: {
                doc_type: docType,
                format: detectedFormat,
                source: document.source
            }
        }
        enrichedChunks.append(enrichedChunk)
    
    return enrichedChunks

function buildBreadcrumb(headers):
    return join(headers, " > ")
```

***

## Data Flow

```
Input Document
    ↓
[STAGE 1: Structure Parsing]
    ├─ Detect format (Markdown/HTML/PDF/plain text)
    └─ Parse hierarchy & create initial chunks
    ↓
Initial Chunks (coarse-grained)
    ↓
[STAGE 2: Embedding Refinement]
    ├─ For large chunks: embed sentences
    ├─ Calculate sentence similarity
    ├─ Detect breakpoints (low similarity)
    └─ Split & merge into final chunks
    ↓
Refined Chunks (semantic boundaries)
    ↓
[STAGE 3: LLM Validation] (optional, critical docs only)
    ├─ Ask LLM to verify boundaries
    └─ Adjust if needed
    ↓
Validated Chunks
    ↓
[STAGE 4: Metadata Enrichment]
    ├─ Add breadcrumbs
    ├─ Add section titles
    └─ Add doc metadata
    ↓
Output: Rich Chunks with Context
    ↓
Store in Vector DB with metadata
```

***

## Configuration Example

```python
config = {
    "maxChunkSize": 1000,              # Merge sub-chunks if < 1000 tokens
    "embeddingThreshold": 0.5,         # Split if similarity < 0.5
    "validateWithLLM": False,           # Only for critical docs
    "embeddingModel": "text-embedding-3-small",
    "llmModel": "gpt-4-turbo"
}

chunks = hybridSemanticChunking(
    document={"content": text, "source": "docs.example.com"},
    docType="technical",
    config=config
)
```

***

## When to Use Each Stage

| Document Type | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| Blog/Blog posts | ✅ | ✅ | ❌ |
| Technical documentation | ✅ | ✅ | ❌ |
| Knowledge base | ✅ | ✅ | ❌ |
| Legal contracts | ✅ | ✅ | ✅ |
| Security policies | ✅ | ✅ | ✅ |
| Plain text (no structure) | ⏭️ | ✅ | ❌ |

***

## Expected Outcomes

- **Better retrieval accuracy:** Chunks are semantically coherent, not arbitrary.
- **Preserved context:** Breadcrumbs and metadata enable richer RAG prompts.
- **Reduced hallucinations:** Complete thoughts stay in one chunk; LLM doesn't piece together fragments.
- **Production-ready:** Balances accuracy, cost, and performance.
