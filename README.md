# 📘 GraphRAG — Complete Session Guide

> **Session Format**: This README is your single source of truth — PPT, Notion, and speaker notes combined.
> Run the `graphrag_session.ipynb` notebook live during the demo.

---

## 🗂️ Table of Contents

1. [What is RAG?](#1-what-is-rag)
2. [What is GraphRAG?](#2-what-is-graphrag)
3. [Core Differences](#3-core-differences)
4. [Why GraphRAG Exists](#4-why-graphrag-exists)
5. [Core Concepts](#5-core-concepts)
6. [End-to-End Pipeline](#6-end-to-end-pipeline)
7. [Environment Setup](#7-environment-setup)
8. [Demo Document](#8-demo-document)
9. [Query Comparisons](#9-query-comparisons)
10. [Proven Advantages of GraphRAG](#10-proven-advantages-of-graphrag)
11. [Proven Limitations of GraphRAG](#11-proven-limitations-of-graphrag)
12. [RAG Advantages (Important!)](#12-rag-advantages)
13. [When to Use What](#13-when-to-use-what)
14. [Ready-to-Speak Lines](#14-ready-to-speak-lines)

---

## 1. What is RAG?

**RAG = Retrieval-Augmented Generation**

| Step | What happens |
|------|-------------|
| 1 | User asks a question |
| 2 | Question is converted to an embedding (vector) |
| 3 | Vector DB finds the most *similar* text chunks |
| 4 | Those chunks are fed to an LLM as context |
| 5 | LLM generates an answer |

```
User Query ──► Embedding ──► Vector Similarity Search ──► Top-K Chunks ──► LLM ──► Answer
```

**Works well for:**
- Direct factual questions ("What is memory in agents?")
- Questions where the answer lives in one paragraph
- Fast, low-latency use cases

> **Simple definition:** RAG finds the most *similar* text to your question and hands it to the LLM.

---

## 2. What is GraphRAG?

**GraphRAG = RAG + Knowledge Graph**

| Step | What happens |
|------|-------------|
| 1 | Documents are parsed for **entities** and **relationships** |
| 2 | A **knowledge graph** is built (nodes = entities, edges = relationships) |
| 3 | User query → entity extraction → graph traversal |
| 4 | Connected knowledge paths are retrieved |
| 5 | LLM reasons over the structured graph context |

```
Documents ──► Entity Extraction ──► Graph DB (Neo4j)
                                          │
User Query ──► Query Entities ──► Graph Traversal ──► Paths ──► LLM ──► Answer
```

**Works well for:**
- Multi-hop reasoning ("How does memory help planning?")
- Questions spanning multiple concepts
- Domains with complex relationships

> **Simple definition:** GraphRAG doesn't just find similar text — it *reasons over connections* between concepts.

---

## 3. Core Differences

| Dimension | RAG | GraphRAG |
|-----------|-----|----------|
| **Retrieval method** | Vector similarity search | Graph relationship traversal |
| **Knowledge format** | Flat text chunks | Structured nodes + edges |
| **Reasoning** | Weak (single-hop only) | Strong (multi-hop, chained) |
| **Explainability** | Low (why these chunks?) | High (shows the path) |
| **Setup complexity** | Simple | Complex |
| **Speed** | Fast | Slower |
| **Best for** | Direct Q&A | Relationship-heavy queries |

---

## 4. Why GraphRAG Exists

**Problems RAG cannot solve:**

1. **Cross-document context gap** — RAG finds chunks but cannot link facts from different documents
2. **Concept linking failure** — Two concepts mentioned separately cannot be connected
3. **Multi-hop query failure** — "A depends on B which uses C" type queries return incomplete answers
4. **Scattered information** — Top-k chunks may be redundant and miss the connecting reasoning

**How GraphRAG solves these:**

1. Connects concepts across the entire document corpus in a unified graph
2. Explicitly stores relationships as edges — no inference needed
3. Graph traversal naturally chains A → B → C reasoning
4. Retrieves compact, structured paths instead of raw text fragments

---

## 5. Core Concepts

```
Entity  ──  Any object of interest
            Examples: "agent", "memory", "tool", "planning"

Node    ──  Entity represented in the graph
            Each unique entity = 1 node

Relationship ── A named connection between two entities
            Examples: "uses", "depends on", "improves"

Edge    ──  Relationship represented in the graph
            Directed: source ──[relationship]──► target

Graph   ──  Collection of nodes + edges
            Represents all knowledge from your documents

Multi-hop ─ Querying chains of relationships
            agent ──uses──► tools ──improve──► reasoning
```

**Example triplets extracted from the demo paper:**

| Source | Relationship | Target |
|--------|-------------|--------|
| agent | uses | tools |
| agent | relies on | memory |
| planning | improves | reasoning |
| memory | stores | past actions |
| tools | extend | agent capabilities |
| planning | decomposes | tasks |

---

## 6. End-to-End Pipeline

### RAG Pipeline

```
1. Load PDF
2. Split into chunks (500 tokens, 50 overlap)
3. Generate embeddings (OpenAI text-embedding-ada-002)
4. Store in ChromaDB (vector store)
5. User query → embed → similarity search → top-3 chunks
6. Chunks → GPT-4o-mini → Answer
```

### GraphRAG Pipeline

```
1. Load PDF
2. Split into chunks
3. For each chunk → LLM extracts (subject, predicate, object) triplets
4. Store triplets in Neo4j (MERGE entities, MERGE relationships)
5. User query → LLM extracts entities from question
6. Neo4j Cypher query → traverse graph from those entities
7. Graph paths → GPT-4o-mini → Structured answer
```

### Full Comparison

```
               ┌─────────────────────────────────┐
               │  DOCUMENT: LLM Agents Paper      │
               └──────────────┬──────────────────┘
                              │
              ┌───────────────┴──────────────────┐
              ▼                                   ▼
    ┌──────────────────┐               ┌──────────────────────┐
    │  RAG PIPELINE    │               │  GraphRAG PIPELINE   │
    │                  │               │                      │
    │  Chunk → Embed   │               │  Chunk → Extract     │
    │  → ChromaDB      │               │  Entities →          │
    │                  │               │  Neo4j Graph         │
    └────────┬─────────┘               └──────────┬───────────┘
             │                                     │
    ┌────────▼─────────┐               ┌──────────▼───────────┐
    │  QUERY           │               │  QUERY               │
    │                  │               │                      │
    │  Embed query →   │               │  Extract entities →  │
    │  Similarity →    │               │  Cypher traversal →  │
    │  Top-K chunks    │               │  Graph paths         │
    └────────┬─────────┘               └──────────┬───────────┘
             │                                     │
             └──────────────┬──────────────────────┘
                            ▼
                    ┌───────────────┐
                    │  GPT-4o-mini  │
                    └───────┬───────┘
                            ▼
                        Answer
```

---

## 7. Environment Setup

### Prerequisites

- Python 3.10+
- `uv` package manager
- Neo4j Aura account (free tier works)
- OpenAI API key

### Fix `uv` PATH Issue

The uv installer puts itself in `~/.local/bin`. If your terminal doesn't find `uv`:

```bash
# Quick fix for current session
source $HOME/.local/bin/env

# Permanent fix — add to ~/.bashrc (run once)
echo '. "$HOME/.local/bin/env"' >> ~/.bashrc
source ~/.bashrc

# Verify
uv --version
```

### Project Initialization

```bash
# Navigate to project folder
cd ~/graphrag-session

# Initialize uv project (creates pyproject.toml)
uv init --no-readme

# Create virtual environment (creates .venv/)
uv venv

# Install all required packages
uv add langchain langchain-openai langchain-community langchain-chroma \
       "chromadb>=0.5,<0.6" neo4j pypdf python-dotenv openai ipykernel requests \
       "onnxruntime>=1.19,<1.20"

# Activate the venv (optional if using uv run / Jupyter kernel)
source .venv/bin/activate
```

### `.env` File (Required)

Create a `.env` file in the project root:

```env
# Neo4j Aura (free tier cloud)
NEO4J_URI=neo4j+s://YOUR_INSTANCE_ID.databases.neo4j.io
NEO4J_USERNAME=YOUR_USERNAME
NEO4J_PASSWORD=YOUR_PASSWORD
NEO4J_DATABASE=YOUR_DATABASE

# OpenAI
OPENAI_API_KEY=sk-proj-...
```

> The `.env` file is already set up in this project. **Never commit it to git.**

### Project Structure After Setup

```
graphrag/
├── .env                      ← API keys & DB credentials
├── .venv/                    ← Virtual environment (uv managed)
├── pyproject.toml            ← Project + dependencies (uv managed)
├── graphrag_session.ipynb    ← Main demo notebook ◄ RUN THIS
├── README.md                 ← This file
├── llm_agents.pdf            ← Auto-downloaded during demo
└── raw_conversation.txt      ← Session source material
```

---

## 8. Demo Document

**Paper used:** [LLM Agents — arXiv 2304.03442](https://arxiv.org/pdf/2304.03442.pdf)

**Why this paper?**
- Rich in multi-hop relationships between AI concepts
- Entities are well-defined and interconnected
- Perfectly demonstrates where GraphRAG surpasses RAG

**Key entities in the paper:**

| Entity | Description |
|--------|-------------|
| Agent | The autonomous LLM-powered system |
| Memory | Short-term and long-term storage of past interactions |
| Tools | External capabilities the agent can invoke (APIs, code, search) |
| Planning | Decomposing complex tasks into steps |
| Reasoning | Drawing conclusions from available information |
| Perception | Processing inputs from the environment |
| Execution | Carrying out planned actions |

**Key relationships extracted:**

```
agent ──────uses──────────► tools
agent ──────relies on─────► memory
planning ───improves──────► reasoning
memory ─────stores────────► past actions
tools ──────extend────────► agent capabilities
planning ───decomposes────► tasks
reasoning ──guides────────► planning
agent ──────perceives─────► environment
```

---

## 9. Query Comparisons

### ✅ Simple Queries — Both RAG and GraphRAG Work

These questions have direct answers in single paragraphs:

| Query | Expected Answer Area |
|-------|---------------------|
| `What is an autonomous agent?` | Definition, role in automation |
| `What are the components of LLM agents?` | Memory, tools, planning, execution |
| `What is memory in agents?` | Short-term, long-term, episodic memory |

> 👀 **Observation for the team:** For these queries, both RAG and GraphRAG produce similar quality answers. RAG might even be slightly faster.

---

### ❌ Multi-hop Queries — GraphRAG Wins

These questions *require connecting multiple concepts*:

| Query | Why RAG fails | Why GraphRAG wins |
|-------|--------------|------------------|
| `How does memory help planning?` | Memory and planning are in separate chunks | Graph path: memory → stores → past actions → guide → planning |
| `How are tools, memory, and planning connected?` | Three separate topics, RAG returns scattered chunks | Graph traversal links all three through agent node |
| `Why do agents need both tools and memory?` | Requires reasoning across document | Graph shows: agent→uses→tools AND agent→relies on→memory |
| `How do tools improve reasoning via planning?` | Multi-hop: tools → planning → reasoning | Graph path chain retrieved directly |

#### Expected RAG output (multi-hop query):
```
"Memory helps agents store information. Planning is the process of 
decomposing tasks. According to the paper, an LLM agent uses memory 
to retain past interactions..."
```
*→ Returns fragments, doesn't connect the two concepts*

#### Expected GraphRAG output (same query):
```
"Based on the knowledge graph:
• agent → relies on → memory
• memory → stores → past actions  
• past actions → inform → planning
• planning → improves → reasoning

Memory helps planning because the agent relies on memory to store 
past actions, which directly inform the planning process. This chain 
shows that without memory, planning would operate without historical 
context, degrading reasoning quality."
```
*→ Shows the path, connects concepts explicitly*

---

## 10. Proven Advantages of GraphRAG

### 1. Multi-hop Reasoning ⭐ (Most Important)
- Connects chain of relationships: A → B → C → D
- RAG can only retrieve "similar" text, it cannot follow relationship chains
- **Demo proof**: "How do tools improve reasoning via planning?" — only GraphRAG answers correctly

### 2. Better Context Retrieval
- Fetches structurally related nodes, not just semantically similar text
- You may miss co-referenced entities in pure vector similarity
- **Demo proof**: Querying "memory" also pulls in "past actions", "experience", "episodic storage"

### 3. Structured Knowledge Representation
- Knowledge is stored as facts, not raw text
- entity → relation → entity format is unambiguous
- No "hallucinated connections" — if the relationship isn't in the graph, it isn't returned

### 4. Explainability
- You can show the team *exactly* why the answer was generated
- Graph paths are interpretable: "We retrieved this answer via these 4 hops"
- RAG cannot explain why it picked chunk #7 over chunk #3

### 5. Cross-Document Reasoning
- Multiple documents can contribute entities/relationships to the same graph
- RAG is fundamentally per-chunk; GraphRAG is globally connected

---

## 11. Proven Limitations of GraphRAG

### 1. Complex Setup
- Requires: entity extraction pipeline + graph database + Neo4j + Cypher queries
- RAG only needs: document loader + vector store + similarity search
- **Bottom line**: 3–4× more code and infrastructure

### 2. Slower Performance
- Entity extraction: LLM call per chunk = slow preprocessing
- Graph traversal: Cypher queries add latency vs direct vector lookup
- **For real-time apps** where response time < 500ms, RAG is better

### 3. Extraction Quality Dependency
- If the LLM extracts wrong entities or wrong relationships → wrong graph → wrong answers
- "agent relies on memory" vs "agent uses memory" — relationship labels matter
- **Garbage in, garbage out**: graph quality = extraction quality

### 4. Overkill for Simple Use Cases
- If your queries are factual and direct → RAG is sufficient and faster
- Adding GraphRAG complexity for FAQ bots, simple document Q&A = unnecessary
- **Rule**: Only use GraphRAG when relationships are the answer

### 5. Graph Schema Design Challenge
- Real-world use: you need to define entity types, relationship types carefully
- Inconsistent entity names ruin traversal ("LLM" vs "large language model" vs "GPT")
- Requires entity normalization and deduplication logic

---

## 12. RAG Advantages

> ⚠️ **Important to cover in session** — don't make it sound like RAG is useless

| Advantage | Detail |
|-----------|--------|
| **Simple to implement** | 10 lines of code with LangChain |
| **Fast response time** | Cosine similarity << graph traversal |
| **Works well for factual queries** | Single-paragraph answers, direct lookups |
| **Less preprocessing** | No entity extraction step needed |
| **Mature ecosystem** | ChromaDB, Pinecone, Weaviate, FAISS — many options |
| **Cheap** | No graph DB infrastructure needed |

---

## 13. When to Use What

```
                    IS THE ANSWER IN ONE PLACE?
                              │
               ┌──────────── YES ─────────────┐
               ▼                               ▼
          Use RAG ✅                  DO RELATIONSHIPS MATTER?
        (Fast, simple)                         │
                               ┌───── YES ─────┴────── NO ──────┐
                               ▼                                 ▼
                        Use GraphRAG ✅                     Use RAG ✅
                    (Multi-hop reasoning)              (Relationships irrelevant)
```

| Use RAG when... | Use GraphRAG when... |
|----------------|----------------------|
| Simple Q&A, FAQ bots | Complex reasoning over domains |
| Direct factual queries | Relationships are the core question |
| Speed is critical | Explainability is required |
| Low-budget projects | Knowledge spans many documents |
| Prototype or POC phase | Production AI with knowledge graphs |

**Real-world examples:**

| Scenario | Recommended |
|----------|-------------|
| Customer support FAQ | RAG |
| Medical diagnosis (drugs + interactions) | GraphRAG |
| Document summarization | RAG |
| Legal knowledge (cases + precedents + laws) | GraphRAG |
| Product search by description | RAG |
| Software architecture reasoning | GraphRAG |
| Resume screening | RAG |
| Supply chain dependency analysis | GraphRAG |

---

## 14. Ready-to-Speak Lines

### Opening
> *"Today I'm showing you GraphRAG — a more powerful alternative to traditional RAG that works by reasoning over relationships, not just finding similar text."*

### Showing Simple Queries
> *"Notice both systems give similar answers for simple queries. RAG is actually faster here. This is important — GraphRAG isn't always better."*

### Showing RAG failure on multi-hop
> *"Watch what happens when I ask a question that requires connecting two concepts that live in different parts of the document..."*

### After showing GraphRAG answer
> *"GraphRAG doesn't just retrieve — it follows the relationship chain and tells you exactly how two concepts are connected."*

### On the setup complexity
> *"Yes, GraphRAG is more complex to build. But in domains like healthcare, legal, or enterprise knowledge management — that complexity is worth it."*

### Closing
> *"RAG finds similar text. GraphRAG finds connected knowledge. Use RAG for direct answers. Use GraphRAG when the relationship IS the answer."*

### Emergency backup (if demo fails)
> *"RAG finds similar text. GraphRAG finds connected knowledge."*

---

## ⚡ Quick Reference Card

```
RAG      = Find → Return
GraphRAG = Find → Traverse → Reason → Return

RAG      = "What does the text say about X?"
GraphRAG = "How are X, Y, and Z connected?"

RAG      = flat chunks in a vector DB
GraphRAG = structured nodes + edges in a graph DB

Use RAG: fast, simple, factual
Use GraphRAG: complex, relational, explainable
```

---

*Session prepared for Simform internal team. Run `graphrag_session.ipynb` for live demo.*
