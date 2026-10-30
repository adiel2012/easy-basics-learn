# Vector Databases: Finding Similar Meanings Fast

> **The one idea in this guide:** a **vector database** stores embeddings (lists of numbers that represent meaning) and quickly finds the ones **closest** to a query vector. To stay fast with millions of vectors, it searches **approximately**: it checks only the most promising places instead of everything.
>
> Everything below explains, demonstrates or deepens that idea.

## 1. In simple words

An ordinary database answers exact questions: "find the customer with ID 4471".

A **vector database** answers a different kind of question: "find the items **most similar** to this one". It works with **embeddings**: vectors where similar meanings sit close together.

```
"How do I get a refund?"  →  [0.8, 0.4]  →  vector database  →  the 3 closest stored vectors
                                                             →  their texts: refund policy, returns form, ...
```

**Analogy: a map of meanings.** Imagine every sentence placed as a dot on a map, with similar sentences close together. A vector database is the map plus a fast way to answer "what's near this spot?" Instead of measuring the distance to every dot in the country, it first finds the right town, then searches only there.

> **Quick check 1.** What kind of question does a vector database answer that an ordinary database is not built for?
>
> *Answers to all checks are in Section 7.*

## 2. What you need to know first

- **[RAG](rag.md)**: the most common reason to use a vector database, and an introduction to embeddings and cosine similarity.

## 3. The key terms

| Term | Meaning |
|---|---|
| Vector / embedding | a list of numbers representing meaning |
| Dimension | how many numbers are in each vector (often hundreds to a few thousand) |
| Similarity / distance | a score for how close two vectors are |
| Nearest neighbour search | finding the stored vectors closest to a query |
| Exact (brute-force) search | comparing the query with **every** stored vector |
| Approximate nearest neighbour (ANN) | a faster search that is almost always, but not always, exactly right |
| Index | an extra structure that makes search fast |
| Recall | the fraction of the true nearest neighbours that the search actually found |
| Metadata | extra fields stored with each vector (date, author, category) |

## 4. How it works

### Three ways to measure closeness

| Measure | Formula | Meaning | Better when |
|---|---|---|---|
| **Dot product** | a · b | direction **and** length | higher |
| **Cosine similarity** | (a · b) ÷ (\|a\| × \|b\|) | direction only | higher |
| **Euclidean distance** | √(sum of (aᵢ − bᵢ)²) | straight-line distance | **lower** |

If all vectors are scaled to length 1 (**normalised**), these three give the **same ranking**. Many embedding models produce normalised vectors for this reason.

### Exact search: simple but slow

Compare the query with every vector, keep the best k.

- 1,000 vectors → 1,000 comparisons: fine.
- 100 million vectors, each with 1,000 numbers → 100 billion multiplications **per query**: too slow for many uses.

### Approximate search: fast and almost always right

The index lets the database skip most vectors. Two popular ideas:

**1. Clustering (IVF: inverted file index).**

- Ahead of time: group the vectors into clusters, and store each cluster's centre (**centroid**).
- At query time: compare the query with the centroids, pick the closest cluster(s), and search **only inside them**.

**2. Graphs (HNSW: hierarchical navigable small world).**

- Ahead of time: connect each vector to some of its near neighbours, like a road network, with "highways" on top levels for long jumps.
- At query time: start somewhere, and repeatedly hop to whichever neighbour is closer to the query, until no neighbour is closer.

Both trade a little accuracy (**recall**) for a lot of speed. Settings like "how many clusters to search" let you choose the balance.

### What the database stores

```
id:        42
vector:    [0.12, -0.53, 0.88, ...]
text:      "Refunds are given within 30 days."
metadata:  { source: "policy.pdf", page: 3, updated: "2026-05-01" }
```

Metadata allows **filtered search**: "the most similar chunks, but only from documents updated this year".

## 5. Try it yourself

### Six vectors, one query

> **Made-up 2-number vectors**, so you can check them by hand.

Stored vectors, in three groups:

| Vector | Value | Group |
|---|---|---|
| A | [1, 0] | right |
| B | [0.9, 0.2] | right |
| C | [0, 1] | top |
| D | [0.1, 0.9] | top |
| E | [−1, 0] | left |
| F | [−0.8, −0.3] | left |

**Query:** q = [0.8, 0.4]

### Exact search: compare with all six

| Vector | Dot product q · v | Euclidean distance | Cosine |
|---|---|---|---|
| A | 0.8 | 0.447 | 0.894 |
| **B** | 0.8 | **0.224** | **0.970** ← closest |
| C | 0.4 | 1.000 | 0.447 |
| D | 0.44 | 0.860 | 0.543 |
| E | −0.8 | 1.844 | −0.894 |
| F | −0.76 | 1.746 | −0.995 |

Check B by hand:

```
dot      = 0.8 × 0.9 + 0.4 × 0.2 = 0.72 + 0.08 = 0.80
distance = √((0.9 − 0.8)² + (0.2 − 0.4)²) = √(0.01 + 0.04) = √0.05 ≈ 0.224
cosine   = 0.80 ÷ (√0.85 × √0.80) = 0.80 ÷ (0.922 × 0.894) ≈ 0.970
```

**Notice:** A and B **tie** on dot product (0.8 each), but cosine and Euclidean both prefer B. The measures don't always agree when vectors have different lengths: A has length 1, B has length 0.922.

### Approximate search: clusters

Centroids (the average of each group):

| Cluster | Centroid | Cosine with q |
|---|---|---|
| right (A, B) | [0.95, 0.1] | **0.936** ← closest |
| top (C, D) | [0.05, 0.95] | 0.494 |
| left (E, F) | [−0.9, −0.15] | −0.956 |

Search only the "right" cluster: compare with A and B → **B wins**. Same answer.

**Comparisons:** 3 centroids + 2 vectors = **5**, instead of 6. That's a tiny saving here, but at scale it's huge:

```
1,000,000 vectors in 1,000 clusters of about 1,000 each
exact:       1,000,000 comparisons
clustered:   1,000 centroids + 1,000 vectors = 2,000 comparisons  (500× fewer)
```

**The risk:** if the true nearest vector sits just across a cluster border, searching one cluster misses it. Searching the 2 or 3 closest clusters fixes most of these misses, at a little extra cost.

> **Quick check 2.** In the clustered example, how many comparisons would you need if you searched the two closest clusters?

## 6. Variations, tips and mistakes

### Do you need a dedicated vector database?

| Option | Good for |
|---|---|
| A plain array and brute force (e.g. NumPy) | up to tens of thousands of vectors, prototypes |
| A search library (e.g. FAISS) | fast search inside your own program |
| A vector extension for a normal database | when your data already lives there |
| A dedicated vector database | large scale, filtering, updates, many users |

### Tips

1. **Use the same embedding model** for stored items and queries. Vectors from different models aren't comparable.
2. **Pick the measure your embedding model was trained for** (often cosine or dot product on normalised vectors).
3. **Measure recall.** Compare approximate results with exact results on a sample of queries.
4. **Store the text and metadata**, not just the vector, so you can show and filter results.
5. **Re-embed everything if you change the embedding model.**

### Common mistakes

| Mistake | What goes wrong | Fix |
|---|---|---|
| Mixing embedding models | similarity scores become meaningless | one model per index |
| Mixing normalised and unnormalised vectors with dot product | long vectors win unfairly | normalise, or use cosine |
| Expecting exact keyword matches | embeddings can miss exact codes or names | add keyword search (hybrid search) |
| Ignoring recall | the "nearest" result is quietly wrong sometimes | test against exact search |
| Building a vector database too early | extra complexity for a small dataset | start with brute force |

## 7. Remember this

### Five essential ideas

1. **Store vectors, search by closeness.** A vector database finds the most similar items, not exact matches.
2. **Three measures:** dot product, cosine, Euclidean. On normalised vectors they agree.
3. **Exact search checks everything**: accurate, but slow at scale.
4. **Approximate search uses an index** (clusters or graphs) to check only promising vectors: much faster, slightly less accurate.
5. **Recall measures that trade-off.** Tune the index to get enough recall at the speed you need.

### Exercises

1. Compute the cosine similarity of q = [0.8, 0.4] and C = [0, 1].
2. Why might two vectors tie on dot product but not on cosine?
3. 10 million vectors are split into 10,000 clusters of about 1,000 each. You search 1 cluster. About how many comparisons is that?
4. An approximate search returns 8 of the true 10 nearest neighbours. What is its recall?

Try the exercises and the quick checks before reading on.

---

### Check your answers

**Quick checks**

1. **"What's most similar to this?"**, a nearest-neighbour search on meaning.
2. 3 centroids + 2 (right) + 2 (top) = **7 comparisons**.

**Exercises**

1. dot = 0.8 × 0 + 0.4 × 1 = 0.4. |q| = √0.8 ≈ 0.894, |C| = 1. Cosine = 0.4 ÷ 0.894 ≈ **0.447**.
2. Dot product depends on **length** too. A longer vector pointing in a slightly worse direction can get the same dot product as a shorter one pointing better. Cosine ignores length.
3. 10,000 centroids + about 1,000 vectors ≈ **11,000 comparisons** (instead of 10 million).
4. 8 ÷ 10 = **0.8 (80%)**.

## 8. Going further

### Where this connects

- **[RAG](rag.md)**: retrieval for LLMs, built on vector search.
- **[Multimodality](multimodality.md)**: image and text embeddings in one shared space can be searched together.
- **[Quantization](quantization.md)**: vectors can be stored in fewer bits to save memory, a common trick in vector databases.

### Short glossary

| Term | Meaning |
|---|---|
| Vector database | stores vectors and finds the most similar ones |
| Nearest neighbour | the stored vector closest to the query |
| ANN | approximate nearest neighbour search |
| IVF | cluster-based index: search only the nearest clusters |
| HNSW | graph-based index: hop toward the query through neighbours |
| Centroid | the average vector of a cluster |
| Recall | share of true nearest neighbours found |
| Normalised vector | a vector scaled to length 1 |

### Learn more

- 📖 [FAISS](https://github.com/facebookresearch/faiss) (Meta): a widely used library for fast vector search
- 📖 [Efficient and Robust Approximate Nearest Neighbor Search Using HNSW Graphs](https://arxiv.org/abs/1603.09320) (Malkov and Yashunin, 2016)
