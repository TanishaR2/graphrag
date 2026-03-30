# RAG → GraphRAG → Hybrid RAG: Complete Concept Guide

> **Session Reference** | Simform Internal Knowledge Session  
> Everything you need to understand RAG, GraphRAG, and Hybrid RAG — theory, trade-offs, limitations, and when to use what.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [What is RAG? (The Beginning)](#2-what-is-rag-the-beginning)
3. [Why GraphRAG Came Into the Picture](#3-why-graphrag-came-into-the-picture)
4. [Why GraphRAG is Useful](#4-why-graphrag-is-useful)
5. [Where GraphRAG Falls Short](#5-where-graphrag-falls-short)
6. [Hybrid RAG — The Best of Both Worlds](#6-hybrid-rag--the-best-of-both-worlds)
7. [The Cost Problem with Hybrid RAG](#7-the-cost-problem-with-hybrid-rag)
8. [Limitations Summary](#8-limitations-summary)
9. [Comparison Table: RAG vs GraphRAG vs Hybrid RAG](#9-comparison-table-rag-vs-graphrag-vs-hybrid-rag)

---

## 1. Architecture Overview

The diagram below describes the full GraphRAG pipeline from raw data to final answer:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GRAPHRAG ARCHITECTURE                               │
│                                                                             │
│  INPUT                                                                      │
│  ┌──────────────┐                                                           │
│  │ Structured   │──► Data Transformation ──────────────────────┐           │
│  │ (CSV, DB)    │                                               │           │
│  └──────────────┘                                              ▼           │
│                        ┌──────────────┐    Named Entity    ┌─────────┐    │
│  ┌──────────────┐      │ Text Chunks  │──► Resolution  ──► │Knowledge│    │
│  │ Unstructured │─────►│              │                     │ Graph   │    │
│  │ (PDF, Text)  │      └──────────────┘                     │ (Neo4j) │    │
│  └──────────────┘             │                             └────┬────┘    │
│                               ▼                                  │         │
│                       Text Embedding Model                        │         │
│                               │                                  │         │
│                               ▼                                  ▼         │
│                        Vector Embeddings    ◄──── Contextually similar     │
│                                                   chunks + connected        │
│                                                   entities as context       │
│                                                                             │
│  QUERY                                                                      │
│  User Prompt ──► [Vector Search + Graph Traversal] ──► Generative Model    │
│                                                               │             │
│                           Compile Final Answer ◄─────────────┘             │
│                                │                                            │
│                           Response to User                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What happens at each stage

| Stage | What it does |
|-------|-------------|
| **Chunking** | Splits large documents into manageable text pieces |
| **Text Embedding Model** | Converts each chunk into a dense vector (semantic fingerprint) |
| **Generative Model (extraction)** | Reads each chunk and pulls out entities + relationships as structured triplets |
| **Named Entity Resolution** | Normalises duplicate entity mentions (`"OpenAI"` and `"open ai"` → same node) |
| **Knowledge Graph** | Stores all entities as nodes, all relationships as edges — queryable via Cypher |
| **Query time** | User question triggers both vector search AND graph traversal; both contexts go to the LLM |
| **Compile Final Answer** | LLM synthesises an answer grounded in retrieved evidence |

---

## 2. What is RAG? (The Beginning)

> **One-liner:** RAG = give the LLM relevant documents at query time instead of baking all knowledge into weights.

### The Problem RAG Solved

Before RAG, LLMs had two options:
- **Pretrain on everything** — expensive, knowledge becomes stale the moment training ends.
- **Fine-tune on your data** — costly, risky, and still doesn't give the model access to documents it hasn't seen.

RAG introduced a third option: **Retrieve, then Generate.**

### How it works (3 steps)

```
1. INGEST
   Document → Split into chunks → Embed each chunk → Store in vector DB (ChromaDB / Pinecone / Weaviate)

2. RETRIEVE
   User question → Embed question → Cosine similarity search → Top-k most relevant chunks

3. GENERATE
   [System Prompt + Retrieved Chunks + User Question] → LLM → Answer
```

### What RAG is good at

- Factual Q&A over a single document or a focused corpus
- "What does this contract say about termination?"
- "Summarise this research paper"
- Grounded answers with source attribution (show which chunk the answer came from)

### The RAG limitation that GraphRAG was built to fix

RAG answers by retrieving **semantically similar** chunks. This means:
- It finds chunks that *look like* the question.
- It **cannot** follow a chain of reasoning: "A causes B, B affects C, therefore A affects C."a
- It **cannot** answer: *"How does memory affect the planning ability of autonomous agents?"* — because the answer requires connecting three different concepts that each live in different chunks.

This is called the **multi-hop reasoning problem.**

---

## 3. Why GraphRAG Came Into the Picture

RAG treats your corpus as a bag of isolated text chunks. It has no concept of how entities relate to each other across chunks. This creates a hard ceiling:

### The two failure modes of pure RAG

**Failure Mode 1 — Fragmented Knowledge**
```
Chunk A: "Memory allows agents to store past experiences."
Chunk B: "Planning requires access to historical context."
Chunk C: "Autonomous agents use planning to achieve goals."

Question: "How does memory help autonomous agents achieve goals?"
RAG retrieves: Maybe Chunk A or Chunk C — never connects all three.
GraphRAG: Traverses Memory → Planning → Autonomous Agent in the graph.
```

**Failure Mode 2 — Cross-Document Reasoning**
```
Document 1 (Company filings):  "Acme Corp acquired Beta Ltd in 2022."
Document 2 (News article):     "Beta Ltd's CEO joined Gamma Inc's board."
Document 3 (Industry report):  "Gamma Inc leads the robotics supply chain."

Question: "Which supply chain does Acme Corp indirectly influence?"
RAG: Cannot connect these three documents.
GraphRAG: Traverses Acme Corp → Beta Ltd → Gamma Inc → robotics supply chain.
```

Microsoft Research published GraphRAG in 2024 precisely to address this: instead of retrieving text, retrieve **structured knowledge** from a graph of entities and their relationships.

---

## 4. Why GraphRAG is Useful

### Core Capability: Explicit Relationship Traversal

GraphRAG stores knowledge as a network:
- **Nodes** = entities (people, concepts, organisations, technologies)
- **Edges** = relationships (`causes`, `uses`, `acquired`, `is_a`, `affects`)

At query time, instead of asking "which chunk is similar to this question?", it asks "which graph paths connect the entities in this question?"

### Where GraphRAG excels

GraphRAG is purpose-built for **global, cross-document reasoning**:

| Query type | Example |
|------------|---------|
| Theme discovery | "What are the common themes across these 500 research papers?" |
| Relationship mapping | "Which companies in this corpus have acquisition relationships?" |
| Narrative synthesis | "Summarise the overall narrative across this entire document set?" |
| Multi-hop chains | "How does concept A connect to concept C through intermediate concept B?" |
| Influence tracing | "Which upstream factors affect this downstream outcome?" |

> It excels when the answer requires connecting information spread across many different documents, because the **graph structure explicitly represents those connections.**

### Structural Advantages

| Advantage | Why it matters |
|-----------|---------------|
| Multi-hop traversal | Follow relationship chains of arbitrary depth |
| Entity deduplication | `"GPT-4"` and `"GPT4"` are the same node — no duplicate retrieval |
| Relationship-aware context | The LLM receives not just text but structured paths: `A → [predicate] → B` |
| Global summarisation | Community detection algorithms can summarise clusters of entities |

---

## 5. Where GraphRAG Falls Short

### Limitation 1 — Cost of Building the Graph

Every document requires **multiple LLM calls** for entity and relationship extraction.

```
10,000 chunks × 1 LLM call per chunk × $0.002/call = $20 just to index
(compare: RAG embedding 10,000 chunks = ~$0.10)
```

For a large corpus, you are essentially **pre-processing all your data with an LLM before any user has asked a single question.** This is expensive and slow.

### Limitation 2 — Lossy Compression

The entity extraction step is a transformation, and **every transformation loses information.**

> Raw text → structured entities/relationships → graph

Edge cases, tone, ambiguous phrasing, and contextual caveats can all get dropped during extraction. The graph is a **simplified version of reality.** If your source material contains subtle nuance (legal documents, medical literature), the graph may strip out exactly the nuance that matters.

### Limitation 3 — Explainability Gap

When GraphRAG retrieves community summaries to answer a question, tracing that answer back to a **specific line in a specific document** is non-trivial.

In regulated industries (legal, finance, healthcare) you need to cite your source precisely. Pure GraphRAG makes that hard.

### Limitation 4 — Entity Normalisation is Hard

The graph is only as good as its entity extraction. If the same entity is referred to in 12 different ways across your corpus, and your normalisation step doesn't catch them all, you get fragmented nodes that should be merged.

### Limitation 5 — Schema Rigidity vs. Flexibility Trade-off

Designing a graph schema upfront means making assumptions about what relationships matter. Get it wrong and you'll need to re-extract and re-ingest the entire corpus.

### Limitation 6 — Vectors are Already Very Good for Local Queries

For simple, direct questions ("What is X?", "When did Y happen?"), vector RAG is:
- Faster (no graph traversal)
- Cheaper (no extraction LLM cost)
- Often equally accurate

GraphRAG is **overkill** for these cases.

---

## 6. Hybrid RAG — The Best of Both Worlds

**The insight:** Don't choose between vectors and graphs. Use both.

```
User Question
     │
     ├──► Vector Search (ChromaDB) ──► Top-k relevant chunks
     │
     └──► Graph Traversal (Neo4j) ──► Entity relationship paths
                    │
                    ▼
          Combine both contexts
                    │
                    ▼
             LLM generates answer
             grounded in BOTH
             semantic similarity
             AND structural relationships
```

### Why Hybrid RAG Works

| Component | What it contributes |
|-----------|-------------------|
| **Vector retrieval** | Captures semantically similar passages even if entities aren't explicitly named |
| **Graph traversal** | Captures explicit multi-hop relationship chains |
| **Combined context** | LLM can see both the "similar text" and the "relationship paths" — coverage is maximised |

### Example

Question: *"How does memory affect the planning ability of autonomous agents, and what tools enable this?"*

- **RAG alone** returns chunk about memory OR chunk about planning — not both, and not the connection.
- **GraphRAG alone** returns graph paths but may miss nuanced text about implementation details.
- **Hybrid RAG** returns: the semantically similar chunks (from vector search) AND the graph path `Memory → [enables] → Planning → [used_by] → Autonomous Agent → [uses] → [tool_A, tool_B]`.

The LLM now has full coverage to answer correctly.

---

## 7. The Cost Problem with Hybrid RAG

Hybrid RAG inherits **all costs** of both systems:

| Cost dimension | Vector RAG | GraphRAG | Hybrid RAG |
|----------------|-----------|---------|-----------|
| Ingestion LLM cost | None | High (extraction per chunk) | High (same as GraphRAG) |
| Embedding cost | Low | Low-Medium | Low-Medium |
| Storage cost | Low (vector DB) | Medium (graph DB license/infra) | High (both) |
| Query latency | Low (single vector lookup) | Medium (graph traversal) | Higher (both searches + merge) |
| Infra complexity | Low | High | Very High |
| Maintenance | Low | High (schema, normalisation) | Very High |

> **Rule of thumb:** Hybrid RAG delivers the highest answer quality, but only makes economic sense when the corpus is large, queries are complex, and the answer quality directly impacts business value (e.g., legal research, biomedical Q&A, enterprise knowledge management).

---

## 8. Limitations Summary

| # | System | Limitation |
|---|--------|-----------|
| 1 | **GraphRAG** | High ingestion cost — multiple LLM calls per chunk |
| 2 | **GraphRAG** | Lossy compression — nuance is lost during entity extraction |
| 3 | **GraphRAG** | Explainability — hard to trace answer back to source line |
| 4 | **GraphRAG** | Entity normalisation failures corrupt the graph |
| 5 | **GraphRAG** | Schema rigidity — wrong schema = full re-ingestion |
| 6 | **RAG** | Multi-hop reasoning failure — can't connect concepts across chunks |
| 7 | **RAG** | No structural understanding — treats docs as a bag of text |
| 8 | **RAG** | Semantic drift — "similar text" ≠ "correct answer" for complex queries |
| 9 | **Hybrid RAG** | All of GraphRAG's ingestion costs |
| 10 | **Hybrid RAG** | Higher query latency (parallel search + merge step) |
| 11 | **Hybrid RAG** | Infra overhead — maintaining both a vector DB and a graph DB |

---

## 9. Comparison Table: RAG vs GraphRAG vs Hybrid RAG

| Factor | Vector RAG | GraphRAG | Hybrid RAG |
|--------|-----------|---------|-----------|
| **Core mechanism** | Semantic similarity (cosine distance in embedding space) | Graph traversal (Cypher / SPARQL hops across entity-relationship graph) | Both: vector similarity + graph traversal, contexts merged |
| **Ingestion cost** | 💚 Low — embed chunks only | 🔴 High — LLM call per chunk for extraction | 🟡 High — same as GraphRAG |
| **Ingestion speed** | 💚 Fast | 🔴 Slow (LLM-bound) | 🔴 Slow |
| **Query latency** | 💚 Low (~50–200 ms) | 🟡 Medium (~200–800 ms, graph traversal) | 🔴 Higher (~500 ms–1.5 s, both lookups + merge) |
| **Answer quality — simple Q&A** | 💚 High | 🟡 Comparable | 💚 High |
| **Answer quality — multi-hop** | 🔴 Weak | 💚 Strong | 💚 Strongest |
| **Answer quality — cross-doc reasoning** | 🔴 Weak | 💚 Strong | 💚 Strongest |
| **Explainability / source citation** | 💚 Easy — return source chunk | 🔴 Hard — answer comes from graph paths across many sources | 🟡 Partial — vector sources citable; graph paths less so |
| **Handling nuance & ambiguity** | 💚 Preserved in raw text | 🔴 Lost during entity extraction | 🟡 Partially preserved via vector component |
| **Entity deduplication** | 🔴 None — same entity in 10 chunks = 10 retrievals | 💚 Built-in — MERGE deduplicates nodes | 💚 Built-in via graph component |
| **Setup complexity** | 💚 Low — pip install + embed | 🔴 High — graph DB, schema design, extraction pipeline | 🔴 Very High — everything from both |
| **Infrastructure cost (monthly)** | 💚 Low (~$10–50 for vector DB) | 🟡 Medium (~$65–200 for Neo4j Aura) | 🔴 High (~$75–250+ for both) |
| **Maintenance effort** | 💚 Low | 🔴 High (schema updates, normalisation, re-ingestion) | 🔴 Very High |
| **Scalability** | 💚 High — vector DBs scale horizontally easily | 🟡 Medium — graph DBs scale vertically, sharding is complex | 🟡 Medium — constrained by graph component |
| **Real-time data updates** | 💚 Easy — embed and insert | 🟡 Medium — extraction + MERGE | 🟡 Medium |
| **Production adoption (2025)** | 💚 Mainstream — default choice at most AI teams | 🟡 Growing — Microsoft, knowledge-intensive use cases | 🟡 Niche — high-value domains |
| **Team skill requirement** | 💚 Any ML engineer | 🔴 ML + graph DB + schema design expertise | 🔴 All of the above |
| **Best for: corpus size** | Small–Medium (< 1M chunks fine) | Large, entity-rich (knowledge bases, encyclopedias) | Large, complex, multi-relational |
| **Best for: query type** | Direct, factual, single-hop | Multi-hop, relational, global summary | Complex, nuanced, cross-document |

### Use Case Recommendation Matrix

| Use Case | Best Choice | Reason |
|----------|-------------|--------|
| Internal FAQ / policy Q&A | **Vector RAG** | Simple direct questions, low budget, fast setup |
| Customer support chatbot | **Vector RAG** | High query volume, latency matters, answers are usually self-contained |
| Legal document review | **Hybrid RAG** | Complex reasoning + need for source citation |
| Medical / clinical Q&A | **Hybrid RAG** or **GraphRAG** | Entity relationships (drug → interaction → condition) critical |
| Enterprise knowledge graph (M&A, org charts) | **GraphRAG** | Data is inherently relational, complex traversal needed |
| Research paper analysis (500+ papers) | **GraphRAG** | Theme discovery, cross-paper entity linking |
| E-commerce product search | **Vector RAG** | Semantic similarity is the primary need |
| Code search / documentation Q&A | **Vector RAG** | Text similarity works well; structure is lexical |
| Competitive intelligence | **Hybrid RAG** | Cross-document entity tracking (companies, products, events) |
| Fraud detection knowledge base | **GraphRAG** | Relationship traversal is the core value (A → transacts_with → B → known_fraudster) |
| Real-time news Q&A | **Vector RAG** | Fast ingestion + low latency outweigh graph benefits |
| Regulated industry reporting | **Hybrid RAG** | Complex reasoning + explainability from vector sources |

---

## Quick Decision Rule

```
Is the answer to the user's question contained in a single passage?
         │
    YES ─┤──► Vector RAG   (fastest, cheapest, good enough)
         │
    NO   └──► Does it require connecting multiple entities/concepts?
                   │
              YES ─┤──► GraphRAG   (if budget allows, corpus is static)
                   │
              NO   └──► Hybrid RAG  (if you need both coverage AND reasoning)
                              │
                         Is query latency < 500 ms a hard requirement?
                              │
                         YES ─┴──► Reconsider: optimise graph traversal depth
                                   or fall back to GraphRAG with caching
```

---

## Practical Numbers at a Glance

| Metric | Vector RAG | GraphRAG | Hybrid RAG |
|--------|-----------|---------|-----------|
| Index 10,000 chunks | ~$0.10 | ~$15–25 | ~$15–25 |
| Query cost (per query) | ~$0.001 | ~$0.005–0.01 | ~$0.008–0.015 |
| Query latency (p50) | 50–200 ms | 200–800 ms | 500–1500 ms |
| Setup time (PoC) | 1–2 hours | 1–2 days | 2–4 days |
| Setup time (production) | 1–3 days | 1–4 weeks | 2–6 weeks |

---

*This document covers the full conceptual landscape. The companion notebook `graphrag_session.ipynb` provides working code for all three approaches.*
