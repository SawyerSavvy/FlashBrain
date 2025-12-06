# RAG Service Storage Architecture

## Overview

The RAG Service supports multiple methods for document input. This document explains the trade-offs and recommendations.

## Supported Methods

### 1. Supabase Storage (RECOMMENDED for Production) ⭐

**How it works:**
- Documents are stored in Supabase Storage buckets
- The RAG service downloads files from buckets when processing
- Files persist and can be referenced multiple times

**Usage:**
```python
process_document(
    supabase_bucket_name="documents",
    supabase_bucket_path="project-123/requirements.pdf",
    title="Project Requirements"
)
```

**Advantages:**
- ✅ **No API timeout issues** - Large files handled efficiently
- ✅ **File persistence** - Documents remain accessible after processing
- ✅ **Scalability** - Better for production workloads
- ✅ **Version control** - Can handle file versioning
- ✅ **Access control** - Supabase Storage policies for security
- ✅ **Asynchronous processing** - Can process files independently
- ✅ **No re-upload needed** - Files already stored, just reference them
- ✅ **CDN support** - Fast global access via Supabase CDN

**Disadvantages:**
- ⚠️ Requires Supabase Storage setup (bucket creation)
- ⚠️ Additional storage costs (but minimal)

**Best for:**
- Production environments
- Large documents (>10MB)
- Documents that need to be processed multiple times
- Multi-tenant applications
- Long-running processing tasks

---

### 2. Direct API Upload (Current Implementation)

**How it works:**
- Files sent directly via HTTP POST to the RAG service
- Files processed immediately and discarded

**Usage:**
```python
process_document(
    file_url="https://example.com/doc.pdf",
    title="Project Requirements"
)
```

**Advantages:**
- ✅ Simple to use
- ✅ No storage setup required
- ✅ Good for quick testing/prototyping

**Disadvantages:**
- ❌ **API timeout risk** - Large files may timeout
- ❌ **No persistence** - Files must be re-uploaded if reprocessing needed
- ❌ **Memory intensive** - Entire file loaded into memory
- ❌ **Not scalable** - Poor for production workloads
- ❌ **Network overhead** - File transferred every time

**Best for:**
- Development/testing
- Small documents (<5MB)
- One-time processing
- Quick prototypes

---

### 3. Local File Path

**How it works:**
- File path provided (must be accessible to RAG service)

**Usage:**
```python
process_document(
    file_path="/path/to/document.pdf",
    title="Project Requirements"
)
```

**Advantages:**
- ✅ Fast (no network transfer)
- ✅ Good for local development

**Disadvantages:**
- ❌ Requires shared filesystem
- ❌ Not suitable for distributed systems
- ❌ Security concerns (file path access)

**Best for:**
- Local development only
- Single-server deployments

---

### 4. Direct Text Content

**How it works:**
- Text content provided directly in request

**Usage:**
```python
process_document(
    content="This is the document content...",
    source_type="text",
    title="Document Title"
)
```

**Advantages:**
- ✅ No file handling needed
- ✅ Good for small text snippets

**Disadvantages:**
- ❌ Limited to text content
- ❌ Size limitations (API payload limits)

**Best for:**
- Small text documents
- Programmatically generated content

---

## Priority Order

The RAG service processes documents in this priority order:

1. **Direct content** (`content` parameter)
2. **Supabase Storage** (`supabase_bucket_name` + `supabase_bucket_path`) ⭐ RECOMMENDED
3. **Local file path** (`file_path`)
4. **External URL** (`file_url`)

---

## Recommended Architecture

### Production Setup

1. **Upload documents to Supabase Storage** (via your application's upload endpoint)
2. **Store metadata** (bucket name, path, document ID) in your database
3. **Process documents** using Supabase Storage references:
   ```python
   process_document(
       supabase_bucket_name="documents",
       supabase_bucket_path=f"{project_id}/{document_id}.pdf",
       title=document_title,
       project_id=project_id,
       client_id=client_id
   )
   ```

### Benefits of This Approach

- **Separation of concerns**: Upload logic separate from processing
- **Retry capability**: Can reprocess documents if needed
- **Audit trail**: Documents stored with metadata
- **Scalability**: Handles large files and high volumes
- **Cost efficiency**: No redundant file transfers

---

## Migration Path

If you're currently using API uploads:

1. **Phase 1**: Add Supabase Storage support (✅ Done)
2. **Phase 2**: Update your upload endpoints to store files in Supabase Storage
3. **Phase 3**: Update RAG service calls to use Supabase Storage references
4. **Phase 4**: Deprecate direct API uploads (optional, keep for backward compatibility)

---

## Example: Complete Workflow

### Step 1: Upload Document to Supabase Storage

```python
from supabase import create_client

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Upload PDF to storage
with open("document.pdf", "rb") as f:
    bucket_path = f"project-123/document-{document_id}.pdf"
    supabase.storage.from_("documents").upload(
        file=f,
        path=bucket_path,
        file_options={"upsert": "true"}
    )
```

### Step 2: Process Document via RAG Service

```python
# Call RAG service with Supabase Storage reference
result = await process_document(
    supabase_bucket_name="documents",
    supabase_bucket_path="project-123/document-abc123.pdf",
    title="Project Requirements",
    project_id="project-123",
    client_id="client-456"
)
```

### Step 3: Search Knowledge Base

```python
# Search for relevant information
results = await search_knowledge_base(
    query="What are the project requirements?",
    project_id="project-123",
    match_count=5
)
```

---

## Performance Comparison

| Method | Large Files (>10MB) | Retry Capability | Scalability | Production Ready |
|--------|-------------------|------------------|-------------|------------------|
| Supabase Storage | ✅ Excellent | ✅ Yes | ✅ Excellent | ✅ Yes |
| API Upload | ❌ Timeout risk | ❌ No | ❌ Poor | ❌ No |
| Local File | ⚠️ Depends | ⚠️ Limited | ❌ Poor | ❌ No |
| Direct Content | ❌ Size limits | ✅ Yes | ⚠️ Limited | ⚠️ Limited |

---

## Conclusion

**For production use, Supabase Storage is the recommended approach** because it:
- Eliminates API timeout issues
- Provides file persistence
- Scales better
- Enables retry and reprocessing
- Supports proper access controls

The implementation supports all methods for backward compatibility, but new integrations should use Supabase Storage.

