# RAG (Retrieval-Augmented Generation): Look It Up, Then Answer

> **The one idea in this guide:** before the model answers, a search step **finds the most relevant documents** and pastes them into the prompt. The model then answers using those documents, not just what it memorised in training.
>
> Everything below explains, demonstrates or deepens that idea.

## 1. In simple words

A language model only knows what was in its training data. It doesn't know your company's documents, last week's news, or your private notes. And when it doesn't know, it may **make something up** (a **hallucination**).

**RAG** fixes this in three steps:

```
1. Retrieve   find the documents that best match the question
2. Augment    paste them into the prompt
3. Generate   the model writes the answer, using those documents
```

**Analogy: a librarian and a writer.** You ask a question. The librarian (retrieval) runs to the shelves and brings back the three most useful pages. The writer (the LLM) reads them and writes your answer. The writer is smart, but without the librarian they'd be answering from memory.

> **Quick check 1.** Does RAG change the model's weights?
>
> *Answers to all checks are in Section 7.*

## 2. What you need to know first

Nothing is required: this guide explains what it uses. For more depth:

- **[In-Context Learning](in_context_learning.md)**: models can use facts given in the prompt. RAG is built on this.
- **[Vector Databases](vector_databases.md)**: where the documents are stored and searched.
- **[Tokenization](tokenization.md)**: why documents must be split into chunks.

## 3. The key terms

| Term | Meaning |
|---|---|
| Retrieval | finding relevant text for a question |
| Chunk | a small piece of a document (for example, a paragraph) |
| Embedding | a list of numbers that represents the **meaning** of a piece of text |
| Embedding model | a model that turns text into an embedding |
| Cosine similarity | a score from −1 to 1 for how similar two embeddings are in direction |
| Top-k retrieval | keeping the k most similar chunks |
| Grounding | basing the answer on the retrieved text |
| Hallucination | a confident answer that isn't supported by facts |

## 4. How it works

RAG has two phases: **preparing** the documents once, and **answering** each question.

### Phase 1: prepare (done once, ahead of time)

```
documents → split into chunks → embed each chunk → store in a vector database
```

1. **Chunk.** Split long documents into small pieces, say a few hundred tokens each. Small pieces can be matched precisely and fit easily into a prompt.
2. **Embed.** An embedding model turns each chunk into a vector. Texts with similar meaning get vectors pointing in similar directions.
3. **Store.** Save each vector with its original text in a **[vector database](vector_databases.md)**.

### Phase 2: answer (for every question)

```
question → embed → find the most similar chunks → build the prompt → LLM → answer
```

1. **Embed the question** with the **same** embedding model.
2. **Search:** compare the question's vector with the stored vectors and keep the top k.
3. **Build the prompt:** instructions + retrieved chunks + question.
4. **Generate:** the LLM answers, using the chunks.

### Measuring similarity: cosine similarity

To compare two vectors a and b:

```
cosine(a, b) = (a · b) / (|a| × |b|)
```

- **a · b** (dot product): multiply matching numbers and add them up.
- **|a|** (length): the square root of the sum of squares.
- The result is **1** when the vectors point the same way, **0** when unrelated, **−1** when opposite.

## 5. Try it yourself

### A tiny knowledge base

> **Made-up embeddings.** Real embeddings have hundreds or thousands of numbers. Ours have 3, so you can check them by hand.

| Chunk | Text | Embedding |
|---|---|---|
| 1 | "Refunds are given within 30 days of purchase." | [1, 0, 1] |
| 2 | "Shipping takes 3 to 5 working days." | [0, 1, 1] |
| 3 | "The shop is open 9 am to 5 pm." | [1, 1, 0] |

**Question:** "Can I get my money back?" → embedding **[1, 0, 0.5]**

### Step 1: score every chunk

Length of the question: |q| = √(1² + 0² + 0.5²) = √1.25 ≈ 1.118
Length of each chunk: √(1 + 0 + 1) = √2 ≈ 1.414 (all three happen to be the same)

| Chunk | Dot product with q | Cosine = dot ÷ (1.414 × 1.118) |
|---|---|---|
| 1 (refunds) | 1·1 + 0·0 + 1·0.5 = **1.5** | 1.5 ÷ 1.581 = **0.949** ← best |
| 2 (shipping) | 0·1 + 1·0 + 1·0.5 = 0.5 | 0.5 ÷ 1.581 = 0.316 |
| 3 (hours) | 1·1 + 1·0 + 0·0.5 = 1.0 | 1.0 ÷ 1.581 = 0.632 |

### Step 2: keep the top 1

The refunds chunk wins with 0.949.

### Step 3: build the prompt

```
Answer the question using only the context below.
If the answer is not in the context, say "I don't know".

Context:
[1] Refunds are given within 30 days of purchase.

Question: Can I get my money back?
```

### Step 4: generate

The model answers something like: **"Yes, refunds are given within 30 days of purchase [1]."**

Notice the question never used the word "refund". Embeddings match **meaning**, not just exact words. That's the advantage over plain keyword search.

> **Quick check 2.** With top-k = 2, which two chunks go into the prompt?

## 6. Variations, tips and mistakes

### Why use RAG instead of fine-tuning?

| | RAG | Fine-tuning |
|---|---|---|
| Add new facts | add documents to the database: instant | retrain |
| Delete or update facts | delete or edit the document | retrain, and old facts may linger |
| Show sources | easy: cite the chunks | hard |
| Teach a style or skill | not its strength | its strength |

A common pattern: **RAG for knowledge, fine-tuning for behaviour**.

### Ways to improve retrieval

- **Hybrid search:** combine embedding search with keyword search. Keywords catch exact names, codes and numbers that embeddings can blur.
- **Re-ranking:** retrieve, say, 20 chunks quickly, then use a slower, more accurate model to pick the best 5.
- **Better chunking:** split at natural boundaries (headings, paragraphs), and let chunks overlap a little so a sentence isn't cut in half.
- **Metadata filters:** search only documents from the right product, date range or department.
- **Query rewriting:** let the LLM rephrase a vague question before searching.

### Common mistakes

| Mistake | What goes wrong | Fix |
|---|---|---|
| Chunks too large | the relevant sentence is diluted by unrelated text | smaller chunks |
| Chunks too small | the chunk lacks the context to make sense | larger chunks, or include the heading |
| Different embedding models for documents and questions | the vectors aren't comparable | always use the same model |
| No "I don't know" instruction | the model guesses when retrieval finds nothing useful | tell it to say when the answer isn't in the context |
| Retrieving too many chunks | more cost, and the key fact can get lost | keep k small; re-rank |
| Trusting retrieved text blindly | documents can be wrong, outdated, or contain hidden instructions | keep sources clean; treat them as data |

### Where RAG fails

RAG is only as good as its retrieval. If the right chunk isn't retrieved, the model can't use it. Many "RAG answered wrongly" problems are really "**retrieval** found the wrong chunk" problems. When debugging, check the retrieved chunks first.

## 7. Remember this

### Five essential ideas

1. **Retrieve, augment, generate.** Find relevant text, paste it into the prompt, then answer.
2. **Embeddings capture meaning.** Similar texts have similar vectors; cosine similarity measures that.
3. **Prepare once, search every time.** Chunk and embed documents ahead of time; embed each question when it arrives.
4. **No retraining.** Updating knowledge is as easy as updating documents.
5. **Retrieval is the weak point.** If the right chunk isn't found, the answer can't be right.

### Exercises

1. Question embedding [0, 1, 0.5]. Using the same three chunks, compute the dot products. Which chunk wins?
2. Why should the prompt say "If the answer is not in the context, say I don't know"?
3. A user asks about order code "ZX-4471". Why might keyword search help here more than embeddings?
4. Name the three steps of RAG.

Try the exercises and the quick checks before reading on.

---

### Check your answers

**Quick checks**

1. **No.** It only changes what's in the prompt.
2. **Refunds (0.949) and hours (0.632).**

**Exercises**

1. Refunds: 0 + 0 + 0.5 = 0.5. Shipping: 0 + 1 + 0.5 = **1.5**. Hours: 0 + 1 + 0 = 1.0. **Shipping wins.** (All three lengths are equal, so the dot products give the same order as cosine.)
2. Without it, when retrieval finds nothing useful, the model may make up an answer.
3. Exact codes have no "meaning" for an embedding to capture; keyword search matches the exact string.
4. **Retrieve, augment, generate.**

## 8. Going further

### Where this connects

- **[Vector Databases](vector_databases.md)**: how millions of embeddings are searched quickly.
- **[In-Context Learning](in_context_learning.md)**: why the model can use text you paste in.
- **[Tool Use](tool_use.md)**: search can also be a tool the model decides to call.
- **[Agents](agents.md)**: agents can search, read and search again.

### Short glossary

| Term | Meaning |
|---|---|
| RAG | retrieval-augmented generation: retrieve, augment, generate |
| Chunk | a small piece of a document |
| Embedding | a vector representing meaning |
| Cosine similarity | how alike two vectors' directions are, from −1 to 1 |
| Top-k | keep the k best matches |
| Hybrid search | embeddings plus keyword search |
| Re-ranking | re-scoring retrieved chunks with a more accurate model |
| Grounding | basing the answer on given sources |

### Learn more

- 📖 [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) (Lewis et al., 2020): the paper that named RAG
- 📖 [Sentence-BERT](https://arxiv.org/abs/1908.10084) (Reimers and Gurevych, 2019): a classic way to make sentence embeddings for search
