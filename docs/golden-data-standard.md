# Golden Data Standard — Agentic RAG for Kubeflow

> **Status:** Draft · **Version:** 0.1.0 · **Date:** 2026-02-11
> **Authors:** Kubeflow Agentic-RAG Maintainers
> **Permalink:** `docs/golden-data-standard.md`

---

## 1 · Purpose

This document defines the **Golden Data Standard (GDS)** — the authoritative
specification for every piece of data that enters the Agentic RAG knowledge
base. Any document, code snippet, issue thread, or architectural diagram that
the RAG agent can retrieve **must** satisfy the criteria below before it is
indexed into ChromaDB.

### 1.1 Design Principles

| Principle | Definition |
|---|---|
| **Authoritative** | Every indexed source is an official, peer-reviewed, or maintainer-approved artifact. Community blog posts and Stack Overflow answers are excluded unless explicitly promoted. |
| **Versioned** | Each document is pinned to a specific release, commit SHA, or calendar snapshot. Queries can be scoped to a version window. |
| **Reproducible** | A given `(source, version)` pair always produces the same set of chunks and embeddings when run through the ingestion pipeline (`pipelines/components/ingest.py`). |
| **Citeable** | Every chunk carries structured metadata sufficient to generate a full citation (URL, title, version, retrieval date). |

---

## 2 · Data Categories

### 2.1 Category Overview

| ID | Category | Example Sources | Refresh Cadence |
|---|---|---|---|
| **C-1** | Official Documentation | `kubeflow/website`, KServe docs, Kubeflow Pipelines SDK reference | On release + weekly delta |
| **C-2** | Source Repositories | `kubeflow/kubeflow`, `kubeflow/pipelines`, `kserve/kserve` | On release tag |
| **C-3** | GitHub Issues & Discussions | Issues labeled `kind/bug`, `kind/feature`; Discussions marked *Answered* | Weekly |
| **C-4** | Architecture & Design Docs | KEPs (Kubeflow Enhancement Proposals), ADRs, design-doc PRs | On merge |
| **C-5** | API References | Proto/OpenAPI specs for KFP v2, KServe v1beta1 | On release |
| **C-6** | Release Artifacts | Release notes, changelogs, migration guides | On release |
| **C-7** | Curated Tutorials | Official Kubeflow examples repo, `kubeflow/examples` | Quarterly review |

### 2.2 Justification Per Category

**C-1 — Official Documentation.**
Primary knowledge source for Kubeflow users. Directly maintained by the
project. Highest trust level.

**C-2 — Source Repositories.**
Code-level context (README files, inline docstrings, Helm charts, Kustomize
overlays) is critical for answering "how does component X work?" questions.
Only files matching the inclusion rules (§3) are indexed — not the entire tree.

**C-3 — GitHub Issues & Discussions.**
Captures edge-case troubleshooting, known bugs, and workarounds that do not yet
appear in official docs. Restricted to *closed/resolved* issues and *answered*
discussions to avoid indexing noise.

**C-4 — Architecture & Design Docs.**
KEPs and ADRs capture the *why* behind design decisions. Essential for deep
technical queries about Kubeflow's internals.

**C-5 — API References.**
Machine-readable API specs are chunked and indexed so the agent can answer
precise parameter-level questions (e.g., "What fields does
`InferenceService.spec.predictor` accept?").

**C-6 — Release Artifacts.**
Release notes and migration guides are high-signal documents that users
frequently need when upgrading.

**C-7 — Curated Tutorials.**
Only tutorials hosted under official Kubeflow GitHub orgs. Third-party
tutorials are excluded to avoid stale or inaccurate guidance.

---

## 3 · Inclusion / Exclusion Rules

### 3.1 Global Inclusion Criteria

A document **is included** if it satisfies **all** of the following:

1. Published or merged in an **official Kubeflow or KServe** GitHub org.
2. Written in a **supported format**: Markdown, reStructuredText, plaintext,
   PDF, YAML/JSON (API specs only), or Python (docstrings only).
3. Has a **deterministic permalink** (URL or `org/repo@ref:path`).
4. Is **not marked deprecated** unless the deprecation notice itself is the
   target (for migration-guide purposes).
5. Language is **English** (i18n corpus is a future workstream).

### 3.2 Global Exclusion Criteria

| Rule | Rationale |
|---|---|
| Auto-generated build/test logs | Noise; not useful to end-users. |
| Vendor-specific managed-Kubeflow docs (e.g., GCP AI Platform Pipelines) | May diverge from upstream; not authoritative for open-source Kubeflow. |
| Draft or unmerged PRs | Not yet approved by maintainers. |
| Community blog posts, Medium articles, Stack Overflow | Unverified accuracy; may become stale. |
| Binary blobs (images, model weights, datasets) | Not textual; cannot be chunked or embedded. |
| Files > 500 KB after text extraction | Likely auto-generated or data dumps. Manually review before inclusion. |
| Issues/Discussions in `open` state without `approved` label | Content still under debate. |

### 3.3 Per-Category File-Pattern Rules

```yaml
# Source-repo inclusion globs (C-2)
include:
  - "README.md"
  - "docs/**/*.md"
  - "**/*.proto"              # gRPC service definitions
  - "config/**/*.yaml"        # Kustomize / Helm values
  - "sdk/python/**/*.py"      # SDK docstrings
  - "CHANGELOG.md"
  - "MIGRATION*.md"

exclude:
  - "**/test/**"
  - "**/testdata/**"
  - "**/vendor/**"
  - "**/node_modules/**"
  - "**/*_generated*"
  - "**/*.pb.go"
```

---

## 4 · Versioning Strategy

### 4.1 Version Model

Every indexed artifact is assigned a **Golden Version Tag (GVT)** of the form:

```
gvt:<major>.<minor>.<patch>[-<qualifier>]
```

| Component | Meaning | Example |
|---|---|---|
| `major` | Kubeflow platform major version | `1` |
| `minor` | Kubeflow platform minor version | `9` |
| `patch` | GDS revision within that platform version | `0` |
| `qualifier` | Optional pre-release label | `rc1` |

**Example:** `gvt:1.9.0` — the first golden dataset cut for Kubeflow 1.9.

### 4.2 Snapshot Mechanics

```
data/
  golden/
    gvt-1.9.0/
      manifest.json        # Full listing of every source + SHA
      docs/                # Fetched raw docs
      repos/               # Extracted repo files
      issues/              # Serialised issue JSON
      metadata/            # Per-file metadata JSONL
    gvt-1.9.1/
      ...
```

Each snapshot is produced by a **Kubeflow Pipeline** (extending
`pipelines/rag_pipeline.py`) that:

1. Clones/fetches sources at pinned refs.
2. Applies inclusion/exclusion filters (§3).
3. Extracts text via the project's existing loaders
   (`src/agentic_rag/ingestion/loader.py`).
4. Writes raw text + metadata to the snapshot directory.
5. Records a `manifest.json` with deterministic hashes.

### 4.3 Immutability Guarantee

Once a GVT snapshot is published:

- **No file may be added, modified, or removed.**
- Corrections are published as a new `patch` increment (e.g.,
  `gvt:1.9.0` → `gvt:1.9.1`).
- The `manifest.json` SHA-256 is stored in the pipeline run metadata for
  auditability.

### 4.4 Retention Policy

| Age | Action |
|---|---|
| Current + previous 2 minor versions | Fully indexed and queryable |
| Older versions | Archived to cold storage; re-indexable on demand |

---

## 5 · Metadata Schema for Citation

Every chunk stored in ChromaDB carries the following metadata fields. The
schema is enforced at ingestion time by the pipeline's `ingest_documents`
component.

### 5.1 Schema Definition

```jsonc
{
  // ── Identity ──────────────────────────────────
  "doc_id":           "sha256:<hash>",          // SHA-256 of raw source file
  "chunk_id":         "sha256:<hash>",          // SHA-256 of chunk text
  "golden_version":   "gvt:1.9.0",             // GDS version tag

  // ── Source ────────────────────────────────────
  "category":         "C-1",                    // Data category (§2)
  "source_org":       "kubeflow",               // GitHub org
  "source_repo":      "website",                // GitHub repo
  "source_ref":       "v1.9.0",                 // Git tag / branch / SHA
  "source_path":      "docs/components/pipelines/overview.md",
  "source_url":       "https://www.kubeflow.org/docs/components/pipelines/overview/",
  "source_format":    "markdown",               // markdown | rst | proto | yaml | python

  // ── Content ───────────────────────────────────
  "title":            "Kubeflow Pipelines Overview",
  "section":          "Architecture",           // H2/H3 heading if applicable
  "language":         "en",

  // ── Chunking ──────────────────────────────────
  "chunk_index":      3,                        // 0-based position in doc
  "chunk_total":      12,                       // Total chunks from this doc
  "chunk_size":       512,                      // Characters
  "chunk_overlap":    64,                       // Overlap with adjacent chunks

  // ── Temporal ──────────────────────────────────
  "source_date":      "2026-01-15",             // Last commit / publish date
  "ingestion_date":   "2026-02-11T14:30:00Z",   // Pipeline execution timestamp
  "expiry_date":      "2026-08-11",             // Auto-review deadline (6 months)

  // ── Provenance ────────────────────────────────
  "pipeline_run_id":  "run-abc123",             // Kubeflow Pipeline run ID
  "pipeline_version": "0.3.0",                  // Version of rag_pipeline.py
  "embedding_model":  "sentence-transformers/all-MiniLM-L6-v2",
  "embedding_dim":    384
}
```

### 5.2 Citation Format

The agent's answer-generation prompt **must** render citations using the
metadata above. The canonical citation format is:

> **[title]** — *source_org/source_repo@source_ref* · `source_path`
> Retrieved: `ingestion_date` · GDS: `golden_version`
> URL: `source_url`

**Example rendered citation:**

> **Kubeflow Pipelines Overview** — *kubeflow/website@v1.9.0* ·
> `docs/components/pipelines/overview.md`
> Retrieved: 2026-02-11 · GDS: gvt:1.9.0
> URL: https://www.kubeflow.org/docs/components/pipelines/overview/

### 5.3 Metadata Validation

The ingestion pipeline **rejects** any chunk where:

| Validation Rule | Action on Failure |
|---|---|
| `doc_id` or `chunk_id` is missing/empty | Hard reject; skip document |
| `source_url` is not a valid URL | Warn; allow if `source_path` is present |
| `golden_version` does not match active GVT | Hard reject |
| `category` is not in `{C-1..C-7}` | Hard reject |
| `ingestion_date` is in the future | Hard reject |

---

## 6 · Update Cadence

### 6.1 Schedule

| Trigger | Categories Affected | Pipeline Stage |
|---|---|---|
| **Kubeflow release** (e.g., 1.9 → 1.10) | All | Full re-ingestion under new GVT |
| **Weekly cron** (Sunday 02:00 UTC) | C-1, C-3 | Delta ingestion — new/changed files only |
| **On-merge webhook** | C-4 | Single-doc ingestion for merged KEP/ADR |
| **Quarterly review** | C-7 | Manual curation pass on tutorials |

### 6.2 Delta Ingestion

To avoid re-processing the entire corpus weekly:

1. Fetch the latest Git log / GitHub API events since the last run.
2. Identify files that are new, modified, or deleted.
3. Remove stale chunks from ChromaDB by `doc_id`.
4. Re-ingest only the changed files.
5. Bump the GVT `patch` component.

### 6.3 Staleness Detection

Chunks with `expiry_date` older than the current date are flagged during
retrieval. The agent's reranker (`src/agentic_rag/retrieval/reranker.py`) applies
a **decay penalty** to stale chunks so fresher content is preferred.

---

## 7 · Integration with Existing Pipeline

The GDS plugs directly into the current codebase:

```
┌────────────────────────────────────────────────────────┐
│  Kubeflow Pipeline (pipelines/rag_pipeline.py)         │
│                                                        │
│  ┌──────────┐   ┌──────────┐   ┌───────────────────┐  │
│  │  Fetch   │──▶│  Filter  │──▶│  Load & Chunk     │  │
│  │  Sources │   │  (§3)    │   │  (loader.py,      │  │
│  │          │   │          │   │   chunker.py)      │  │
│  └──────────┘   └──────────┘   └────────┬──────────┘  │
│                                         │              │
│  ┌──────────────────┐   ┌───────────────▼──────────┐  │
│  │  Write Manifest  │◀──│  Embed & Store           │  │
│  │  (manifest.json) │   │  (embedder.py → Chroma)  │  │
│  └──────────────────┘   └──────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

### 7.1 Loader Changes

The existing `load_directory()` function in
`src/agentic_rag/ingestion/loader.py` accepts a `glob` parameter. GDS
enforcement adds a **pre-filter step** that applies inclusion/exclusion rules
(§3.3) before passing paths to the loader.

### 7.2 Metadata Injection

After chunking via `chunk_documents()` in
`src/agentic_rag/ingestion/chunker.py`, a new `enrich_metadata()` function
stamps every `Document.metadata` dict with the GDS fields (§5.1).

### 7.3 ChromaDB Collection Naming

Collections follow the pattern:

```
agentic_rag__gvt_<major>_<minor>_<patch>
```

Example: `agentic_rag__gvt_1_9_0`. The `settings.chroma_collection` value in
`src/agentic_rag/config.py` is dynamically set by the pipeline based on the
active GVT.

---

## 8 · Quality Gates

Before a GVT snapshot is promoted to **active** (queryable by the agent), it
must pass:

| Gate | Criteria | Automated? |
|---|---|---|
| **Completeness** | ≥ 95 % of expected source files present vs. manifest | Yes |
| **Deduplication** | < 2 % chunk-level duplication (cosine similarity > 0.97) | Yes |
| **Schema Compliance** | 100 % of chunks pass metadata validation (§5.3) | Yes |
| **Retrieval Smoke Test** | Top-5 results for 20 canonical queries include at least 1 relevant chunk (evaluated via `pipelines/components/evaluate.py`) | Yes |
| **Human Spot-Check** | Maintainer reviews 10 random citations for accuracy | No |

---

## 9 · Security & Access Control

| Concern | Mitigation |
|---|---|
| Sensitive data in issues (API keys, PII) | Pre-ingestion regex scrub for known secret patterns |
| License compliance | Only index repos with OSI-approved licenses (Apache-2.0, MIT) |
| Data exfiltration via agent | Agent responses are grounded; no raw document regurgitation |

---

## 10 · Glossary

| Term | Definition |
|---|---|
| **GDS** | Golden Data Standard — this specification |
| **GVT** | Golden Version Tag — semver label for a data snapshot |
| **KEP** | Kubeflow Enhancement Proposal |
| **ADR** | Architecture Decision Record |
| **Delta ingestion** | Incremental update that processes only changed files |
| **Chunk** | A text segment produced by `RecursiveCharacterTextSplitter` (default 512 chars, 64 overlap) |
| **Manifest** | `manifest.json` — deterministic listing of all source files and their SHA-256 hashes |

---

## Appendix A · Canonical Source Table

| Source | URL / Identifier | Category | License |
|---|---|---|---|
| Kubeflow website | `kubeflow/website` | C-1 | Apache-2.0 |
| Kubeflow Pipelines | `kubeflow/pipelines` | C-2, C-5 | Apache-2.0 |
| Kubeflow manifests | `kubeflow/manifests` | C-2 | Apache-2.0 |
| KServe | `kserve/kserve` | C-2, C-5 | Apache-2.0 |
| KServe website | `kserve/website` | C-1 | Apache-2.0 |
| Kubeflow examples | `kubeflow/examples` | C-7 | Apache-2.0 |
| KFP SDK reference | `kubeflow/pipelines/sdk` | C-5 | Apache-2.0 |
| Kubeflow community | `kubeflow/community` (KEPs) | C-4 | Apache-2.0 |

## Appendix B · Metadata JSONL Example

```json
{"doc_id":"sha256:a1b2c3...","chunk_id":"sha256:d4e5f6...","golden_version":"gvt:1.9.0","category":"C-1","source_org":"kubeflow","source_repo":"website","source_ref":"v1.9.0","source_path":"docs/components/pipelines/overview.md","source_url":"https://www.kubeflow.org/docs/components/pipelines/overview/","source_format":"markdown","title":"Kubeflow Pipelines Overview","section":"Architecture","language":"en","chunk_index":0,"chunk_total":12,"chunk_size":512,"chunk_overlap":64,"source_date":"2026-01-15","ingestion_date":"2026-02-11T14:30:00Z","expiry_date":"2026-08-11","pipeline_run_id":"run-abc123","pipeline_version":"0.3.0","embedding_model":"sentence-transformers/all-MiniLM-L6-v2","embedding_dim":384}
```

---

*This document is itself subject to the GDS versioning policy. Changes require
a PR review from at least two maintainers.*
