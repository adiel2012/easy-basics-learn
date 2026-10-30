# Multimodality: Models That See, Hear and Read

> **The one idea in this guide:** a **multimodal** model handles more than one kind of data, such as text, images and audio. The trick is to turn every kind of input into the **same thing: a sequence of vectors (tokens)** that one transformer can process together.
>
> Everything below explains, demonstrates or deepens that idea.

## 1. In simple words

A **modality** is a kind of data: text, images, audio, video.

- A **text-only** model reads and writes text.
- A **multimodal** model can, for example, look at a photo and answer a question about it, or listen to speech and reply.

```
[photo of a cat on a sofa]  +  "What animal is this?"   →   "A cat, sitting on a sofa."
```

How? A transformer doesn't really "read words". It processes **vectors**. Text is turned into vectors by tokenization and embedding. If we can also turn an image into vectors, the transformer can process both, side by side.

**Analogy: a translator at a meeting.** Everyone speaks a different language (pictures, sounds, words), so a translator turns each one into a shared language before the discussion. The **encoders** are the translators; the transformer is the meeting.

> **Quick check 1.** What does a multimodal model turn every input into, so one transformer can process it?
>
> *Answers to all checks are in Section 7.*

## 2. What you need to know first

- **[Tokenization](tokenization.md)**: how text becomes tokens and vectors.
- **[Vision Transformer (ViT)](vit.md)**: how an image is cut into patches that act like tokens. **Very useful here.**
- **[Attention Blocks](attention_blocks.md)**: how tokens share information.

## 3. The key terms

| Term | Meaning |
|---|---|
| Modality | a kind of data: text, image, audio, video |
| Multimodal model | a model that handles more than one modality |
| Encoder | a network that turns one modality into vectors |
| Patch | a small square of an image, treated like a token |
| Projection | a small layer that converts encoder vectors to the size the LLM expects |
| Shared embedding space | a space where, for example, an image and its caption get similar vectors |
| Contrastive learning | training that pulls matching pairs together and pushes non-matching pairs apart |

## 4. How it works

### Step 1: turn each modality into tokens

| Modality | How it becomes vectors |
|---|---|
| **Text** | split into tokens → embedding table |
| **Image** | cut into patches → a vision encoder (often a [ViT](vit.md)) → one vector per patch |
| **Audio** | split into short time slices (e.g. a spectrogram) → an audio encoder → vectors |
| **Video** | a series of images (frames), often plus audio |

**How many tokens is an image?** With a ViT using 16 × 16 pixel patches on a 224 × 224 image:

```
patches per side = 224 ÷ 16 = 14
total patches    = 14 × 14 = 196 image tokens
```

So that small image costs about as many tokens as a long paragraph. Bigger images mean more patches, and more tokens.

### Step 2: combine them

A common design for a model that **reads** images and **writes** text:

```
image → vision encoder → projection ─┐
                                     ├─▶ one sequence of vectors → LLM → text answer
text  → tokenizer → embeddings ──────┘
```

The **projection** is a small layer that converts the vision encoder's vectors into the same width as the LLM's text embeddings. After that, the LLM simply sees a sequence: some vectors came from the image, some from text. **Attention** lets text tokens look at image tokens, and vice versa.

The output is still produced by the same next-token loop as in [LLM Inference](llm_inference.md).

### Matching images and text: shared embedding space

Another important idea, made famous by **CLIP** (2021): train an image encoder and a text encoder **together**, so that a photo and its caption get **similar vectors**.

Training uses many image-caption pairs. For each batch:

- **matching pairs** (photo of a dog + "a dog"): make their vectors more similar,
- **non-matching pairs** (photo of a dog + "a car"): make them less similar.

This is **contrastive learning**. Afterwards you can search images with text, or classify an image by comparing it to captions like "a photo of a cat" and "a photo of a dog".

## 5. Try it yourself

### Match images to captions

> **Made-up vectors.** Real ones have hundreds of numbers. Ours have 2.

| Item | Vector |
|---|---|
| Image 1 (a dog) | [1, 0.2] |
| Image 2 (a car) | [0.1, 1] |
| Caption A: "a dog" | [0.9, 0.1] |
| Caption B: "a car" | [0.2, 1] |

Use **cosine similarity**: (a · b) ÷ (|a| × |b|). Higher means more similar.

**Image 1 vs Caption A:**

```
dot     = 1 × 0.9 + 0.2 × 0.1 = 0.92
lengths = √(1 + 0.04) × √(0.81 + 0.01) = 1.020 × 0.906 = 0.924
cosine  = 0.92 ÷ 0.924 ≈ 0.996
```

The full table:

| | Caption A ("a dog") | Caption B ("a car") |
|---|---|---|
| **Image 1** (dog) | **0.996** | 0.385 |
| **Image 2** (car) | 0.209 | **0.995** |

Each image is most similar to its correct caption: the diagonal is high, the rest is low. That is exactly what contrastive training aims for.

> **Quick check 2.** A 448 × 448 image is cut into 16 × 16 patches. How many image tokens is that?

## 6. Variations, tips and mistakes

### Ways to build multimodal models

| Approach | Idea | Example uses |
|---|---|---|
| **Encoder + LLM** | a separate vision encoder feeds an LLM through a projection | describing images, reading charts |
| **Natively multimodal** | one model trained from the start on mixed text, images and audio | fluid conversation across modalities |
| **Contrastive (CLIP-style)** | two encoders, one shared space; no text generation | image search, zero-shot image classification |
| **Generation models** | produce images or audio from text | text-to-image, text-to-speech |

### Input vs output

"Multimodal" can mean different things:

- **multimodal input**: it can *read* images or audio (most common),
- **multimodal output**: it can also *produce* images or audio.

A model that can look at a photo can't necessarily create one.

### Common mistakes and limits

| Issue | What happens | Tip |
|---|---|---|
| Images cost many tokens | slower and more expensive; fills the context | resize to what's needed |
| Small text or fine detail | low-resolution patches can blur it | use higher resolution, or crop to the important part |
| Counting and exact positions | models often miscount objects or confuse left/right | ask for step-by-step description, or verify |
| Hallucinating details | the model "sees" things that aren't there | ask it to say what it's unsure of; check important facts |
| Text inside images | an image can contain written instructions (a form of prompt injection) | treat image text as data |

## 7. Remember this

### Five essential ideas

1. **A modality is a kind of data**; multimodal models handle several.
2. **Everything becomes vectors.** Text via embeddings, images via patches and a vision encoder, audio via an audio encoder.
3. **A projection bridges the gap**, so the LLM sees one mixed sequence and attention links them.
4. **Shared spaces match meaning across modalities:** contrastive training puts an image and its caption close together.
5. **Images are expensive tokens**, and fine detail, counting and positions are common weak spots.

### Exercises

1. A 336 × 336 image, 14 × 14 pixel patches. How many patches?
2. Using the vectors above, compute the dot product of Image 2 and Caption B.
3. Why is a projection layer needed between a vision encoder and an LLM?
4. Can a model with multimodal **input** always generate images? Why or why not?

Try the exercises and the quick checks before reading on.

---

### Check your answers

**Quick checks**

1. **A sequence of vectors** (tokens), the same form for every modality.
2. 448 ÷ 16 = 28 per side, so 28 × 28 = **784 tokens**.

**Exercises**

1. 336 ÷ 14 = 24 per side, so 24 × 24 = **576 patches**.
2. 0.1 × 0.2 + 1 × 1 = **1.02**.
3. The encoder's vectors usually have a different width and "meaning" from the LLM's embeddings; the projection maps them into the LLM's space.
4. **No.** Reading images and producing images are different abilities; a model may only have the first.

## 8. Going further

### Where this connects

- **[Vision Transformer (ViT)](vit.md)**: the most common image encoder.
- **[Vector Databases](vector_databases.md)**: store image and text embeddings for search.
- **[Agents](agents.md)**: computer-use agents read screenshots.
- **[Mixture of Experts](mixture_of_experts.md)**: another way large models scale.

### Short glossary

| Term | Meaning |
|---|---|
| Modality | a type of data |
| Encoder | turns one modality into vectors |
| Patch | a small square of an image |
| Projection | maps encoder vectors into the LLM's embedding space |
| CLIP | a model with matched image and text encoders |
| Contrastive learning | pull matching pairs together, push others apart |
| Spectrogram | a picture of sound: frequencies over time |

### Learn more

- 📖 [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) (Radford et al., 2021)
- 📖 [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485) (Liu et al., 2023): connecting a vision encoder to an LLM with a projection
