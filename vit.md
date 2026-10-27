# Vision Transformer (ViT)

## 1. In simple words

A **Vision Transformer (ViT)** is a transformer that looks at **images** instead of text.

Transformers were built for text, where the input is a sequence of tokens. A ViT turns an image into a sequence too:

1. **Cut** the image into small square pieces called **patches**.
2. **Turn** each patch into a vector, just like a word token becomes an embedding.
3. **Feed** that sequence of vectors into a normal transformer.

**Analogy: a jigsaw puzzle.** Imagine cutting a photo into puzzle pieces and laying them out in a row. Each piece on its own shows only a small part of the picture. By comparing every piece with every other piece, the model works out what the whole picture shows. That comparing is exactly what attention does.

That's why the original paper is called *"An Image is Worth 16x16 Words"*: each 16×16-pixel patch plays the role of one word.

```
image  →  patches  →  patch vectors  →  transformer  →  "cat"
```

## 2. What you need to know first

- **[Tokenization](tokenization.md)**: how text becomes a sequence of tokens and embeddings. ViT does the same thing for images.
- **[Attention Blocks](attention_blocks.md)**: how tokens share information. ViT uses exactly these blocks.
- **Pixels**: a digital image is a grid of numbers. A grayscale image has one number per pixel. A color image has three (red, green, blue), called **channels**.

## 3. The key terms

| Term | Meaning | Example (ViT-Base) |
|---|---|---|
| **Patch** | a small square piece of the image | 16 × 16 pixels |
| **P** | patch size (width and height in pixels) | 16 |
| **C** | number of channels | 3 (red, green, blue) |
| **N** | number of patches | 196 |
| **Patch embedding** | the vector a patch is turned into | 768 numbers |
| **d** | model width (length of each vector) | 768 |
| **[CLS] token** | an extra learned vector added at the start, used to classify the image | |
| **Position embedding** | a learned vector that tells the model where each patch was | |
| **Classification head** | the final layer that turns the [CLS] vector into class scores | |

## 4. How it works

### The pipeline at a glance

```
1. Image            H × W × C
2. Cut into patches  N patches, each P × P × C
3. Flatten           N × (P·P·C)
4. Project           N × d          (patch embeddings)
5. Add [CLS]         (N+1) × d
6. Add positions     (N+1) × d      ← this is X, the transformer input
7. Transformer       (N+1) × d      (encoder blocks, no causal mask)
8. Take [CLS] row    1 × d
9. Classify          1 × number of classes
```

The number of patches is:

```
N = (H / P) × (W / P)
```

### Step by step

**Steps 2–3: cut and flatten.** The image is cut into a grid of patches, which are read row by row, left to right. Each patch is flattened into one long list of numbers, with P × P × C numbers in it.

**Step 4: project.** Each flattened patch is multiplied by a learned matrix **W_E**, of shape (P·P·C) × d. This plays the same role as the embedding table for text: it turns each piece into a d-sized vector.

**Step 5: add a [CLS] token.** One extra learned vector is put at the front of the sequence. It doesn't come from any patch. Its job is to collect information from all the patches through attention, so that at the end it summarizes the whole image.

**Step 6: add position embeddings.** Attention on its own ignores order, so without help the model wouldn't know which patch was top-left and which was bottom-right. A learned position vector is added to each row.

**Step 7: transformer encoder.** The sequence goes through a stack of standard transformer blocks, the same ones explained in the [Attention Blocks](attention_blocks.md) guide. There is **no causal mask**: every patch can attend to every other patch, because nothing is being predicted "in order".

**Steps 8–9: classify.** Only the [CLS] row of the output is used. A small classification head turns it into one score per class, and softmax turns the scores into probabilities.

## 5. Try it yourself: a tiny ViT by hand

> **Note:** This is a toy example with tiny sizes and made-up weights, chosen so you can check every number by hand. In a real ViT, all of these weights are learned during training.

### Sizes

| Symbol | Toy value | Real ViT-Base |
|---|---|---|
| Image H × W × C | 4 × 4 × 1 (grayscale) | 224 × 224 × 3 |
| Patch size P | 2 | 16 |
| Patches N | (4/2) × (4/2) = **4** | (224/16) × (224/16) = 196 |
| Flattened patch P·P·C | 2·2·1 = **4** | 16·16·3 = 768 |
| Model width d | **3** | 768 |
| Sequence length N + 1 | **5** | 197 |

### Step 1: the image

A 4 × 4 grayscale image (each number is a pixel's brightness):

```
[[1, 0 | 2, 2],
 [0, 1 | 2, 2],
 ------+------
 [0, 0 | 1, 1],
 [3, 3 | 1, 0]]
```

The lines show how it will be cut into four 2 × 2 patches.

### Steps 2–3: cut and flatten

Read each patch row by row and write it as one list:

| Patch | Position | Pixels | Flattened |
|---|---|---|---|
| P1 | top-left | `[[1,0],[0,1]]` | `[1, 0, 0, 1]` |
| P2 | top-right | `[[2,2],[2,2]]` | `[2, 2, 2, 2]` |
| P3 | bottom-left | `[[0,0],[3,3]]` | `[0, 0, 3, 3]` |
| P4 | bottom-right | `[[1,1],[1,0]]` | `[1, 1, 1, 0]` |

Stacked, this is a matrix of shape N × (P·P·C) = **4 × 4**.

### Step 4: project to width d

The learned projection W_E has shape (P·P·C) × d = **4 × 3**:

```
W_E = [[1, 0, 0],
       [0, 1, 0],
       [0, 0, 1],
       [1, 1, 1]]
```

Multiply each flattened patch by W_E: (1 × **4**)(**4** × 3) → 1 × 3.

Full calculation for P1 = [1, 0, 0, 1]:

```
column 1: 1·1 + 0·0 + 0·0 + 1·1 = 2
column 2: 1·0 + 0·1 + 0·0 + 1·1 = 1
column 3: 1·0 + 0·0 + 0·1 + 1·1 = 1
→ [2, 1, 1]
```

The others, the same way:

- P2 = [2, 2, 2, 2] → [4, 4, 4]
- P3 = [0, 0, 3, 3] → [3, 3, 6]
- P4 = [1, 1, 1, 0] → [1, 1, 1]

```
patch embeddings = [[2, 1, 1],
                    [4, 4, 4],
                    [3, 3, 6],
                    [1, 1, 1]]    (4 × 3)
```

(Real models also add a learned bias here. We leave it out to keep the numbers simple.)

### Step 5: add the [CLS] token

The [CLS] token is a learned vector of width d. Say it is [0, 1, 0]. Put it at the front:

```
          [[0, 1, 0],    ← [CLS]
           [2, 1, 1],    ← P1
           [4, 4, 4],    ← P2
           [3, 3, 6],    ← P3
           [1, 1, 1]]    ← P4        (5 × 3)
```

### Step 6: add position embeddings

The position embeddings are learned too, one row per position, shape (N+1) × d = **5 × 3**:

```
pos = [[0, 0, 0],
       [1, 0, 0],
       [0, 1, 0],
       [0, 0, 1],
       [1, 1, 1]]
```

Add them row by row. The shapes match, (5 × 3) + (5 × 3) → 5 × 3:

```
X = [[0, 1, 0],    ← [CLS]: [0,1,0] + [0,0,0]
     [3, 1, 1],    ← P1:    [2,1,1] + [1,0,0]
     [4, 5, 4],    ← P2:    [4,4,4] + [0,1,0]
     [3, 3, 7],    ← P3:    [3,3,6] + [0,0,1]
     [2, 2, 2]]    ← P4:    [1,1,1] + [1,1,1]     (5 × 3)
```

This X is the transformer input: **n = 5 tokens, width d = 3**. From here, it's exactly the setup of the [Attention Blocks](attention_blocks.md) guide.

### Step 7: the transformer encoder

X goes through the encoder blocks, and the shape stays 5 × 3 the whole way. Let's calculate by hand how the [CLS] token gathers information from the patches, using one tiny attention head. (The [Attention Blocks](attention_blocks.md) guide explains every part of attention in more detail.)

> **Note:** To keep the arithmetic small, this head uses d_k = d_v = 1 and simple, made-up weights. Real ViT heads are wider, and their weights are learned.

#### Step 7A: calculate Q, K and V

Choose projection matrices, each of shape d × 1 = **3 × 1**:

```
W_Q = W_K = [[0],      W_V = [[1],
             [1],             [0],
             [0]]             [0]]
```

Then Q = X W_Q, K = X W_K and V = X W_V, each (5 × **3**)(**3** × 1) → **5 × 1**.

Multiplying by W_Q (or W_K) just picks the **second** number of each row of X. Multiplying by W_V picks the **first**:

```
            Q = K     V
[CLS]   [[  1  ],  [[ 0 ],
P1       [  1  ],   [ 3 ],
P2       [  5  ],   [ 4 ],
P3       [  3  ],   [ 3 ],
P4       [  2  ]]   [ 2 ]]
```

#### Step 7B: the [CLS] attention scores

The attention formula is:

```
A = softmax( Q Kᵀ / √d_k )
```

Here d_k = 1, so √d_k = 1 and there's nothing to divide by.

The full attention matrix would be 5 × 5, one row per token. This is a **partial calculation**: we only compute the **first row**, for [CLS], because that's the row used for classification. The other rows are computed the same way. Its query is q = 1, so its score against each key is just 1 × that key:

```
          [CLS]  P1   P2   P3   P4
scores = [  1,    1,   5,   3,   2 ]
```

Because ViT uses **no causal mask**, [CLS] scores every patch, including the ones that come after it in the sequence.

#### Step 7C: softmax by hand

Softmax turns scores into weights that sum to 1:

```
softmax(sᵢ) = e^(sᵢ) / Σⱼ e^(sⱼ)
```

| Token | Score | e^score | Weight (e^score / 181.324) |
|---|---|---|---|
| [CLS] | 1 | 2.718 | 0.0150 |
| P1 | 1 | 2.718 | 0.0150 |
| P2 | 5 | 148.413 | 0.8185 |
| P3 | 3 | 20.086 | 0.1108 |
| P4 | 2 | 7.389 | 0.0408 |
| **Sum** | | **181.324** | **1.0000** (to rounding) |

P2 gets most of the attention, because its key gave the highest score.

#### Step 7D: combine the values

The output for [CLS] is the weighted sum of all the values:

```
o_CLS = 0.0150·0 + 0.0150·3 + 0.8185·4 + 0.1108·3 + 0.0408·2
      ≈ 0     + 0.045   + 3.274   + 0.332   + 0.082
      ≈ 3.733
```

Using the exact (unrounded) softmax weights also gives ≈ 3.733.

**What just happened?** The [CLS] token compared its query with every patch's key, turned the scores into weights, and took a weighted mix of the patches' values. That's how it gathers information from the whole image. Here it listened mostly to P2.

#### From one head to a full transformer block

Our output, 3.733, is a **single number (a scalar)**, because this toy head has d_v = 1. It is **not yet** a full 3-number token representation. Here's how a complete block turns head outputs back into width d = 3, using the toy sizes:

| Step | What happens | Shape |
|---|---|---|
| 1. Heads | Run, say, 3 heads in parallel, each like ours (d_v = 1), each with its own W_Q, W_K, W_V | 3 outputs of 5 × 1 |
| 2. Concatenate | Put the head outputs side by side | 5 × 3 |
| 3. Project | Multiply by an output matrix W_O (3 × 3) | 5 × 3 |
| 4. Residual | Add X back: X + result | 5 × 3 |
| 5. Feed-forward | Normalize, then pass each row through a small network, and add the residual again | 5 × 3 |

The block's output has the same shape as X, 5 × 3, so blocks can be stacked. A real ViT-Base stacks 12 of them. The [CLS] value we calculated, 3.733, is one entry in step 1. The central idea, [CLS] gathering information through attention, is exactly what you just computed.

#### After the encoder

> ⚠️ **Separate example.** Working through all the blocks by hand would take pages, so the classification below uses a **made-up encoder output**. It is **not** calculated from Steps 7A–7D. Treat it as a new starting point, not a continuation.

Suppose that after the full encoder, the [CLS] row is:

```
z = [1.0, 0.5, -0.5]    (1 × 3)
```

### Steps 8–9: classify

Say there are 2 classes, **cat** and **dog**. The classification head W_head has shape d × classes = **3 × 2**:

```
           cat  dog
W_head = [[ 1,   0],
          [ 0,   1],
          [ 1,  -1]]
```

Scores (called **logits**): z · W_head, (1 × **3**)(**3** × 2) → **1 × 2**

```
cat: 1.0·1 + 0.5·0 + (−0.5)·1    = 0.5
dog: 1.0·0 + 0.5·1 + (−0.5)·(−1) = 1.0
```

Softmax turns the scores into probabilities:

```
e^0.5 ≈ 1.649,  e^1.0 ≈ 2.718,  sum ≈ 4.367

cat: 1.649 / 4.367 ≈ 0.38
dog: 2.718 / 4.367 ≈ 0.62
```

The model says **dog, with 62% probability**.

(The image and all the weights were made up, so "dog" means nothing here. The point is to see the mechanics: how a vector becomes class probabilities.)

### The full shape story

| Step | Toy shape | ViT-Base shape |
|---|---|---|
| Image | 4 × 4 × 1 | 224 × 224 × 3 |
| Flattened patches | 4 × 4 | 196 × 768 |
| Patch embeddings | 4 × 3 | 196 × 768 |
| + [CLS] | 5 × 3 | 197 × 768 |
| + positions = X | 5 × 3 | 197 × 768 |
| After encoder | 5 × 3 | 197 × 768 |
| [CLS] row | 1 × 3 | 1 × 768 |
| Class scores | 1 × 2 | 1 × 1000 (ImageNet classes) |

## 6. Variations and common mistakes

### Real ViT sizes (optional reading)

| Model | Layers | Width d | Heads | Parameters |
|---|---|---|---|---|
| ViT-Base | 12 | 768 | 12 | about 86 million |
| ViT-Large | 24 | 1024 | 16 | about 307 million |
| ViT-Huge | 32 | 1280 | 16 | about 632 million |

Names like **ViT-B/16** mean "Base size, 16 × 16 patches". Smaller patches mean more patches, so the model sees more detail but needs more computation.

### ViT vs. CNNs

Before ViT, most image models were **CNNs** (convolutional neural networks). The difference in one sentence:

- A **CNN** looks at small neighborhoods first and builds up to the whole image layer by layer.
- A **ViT** lets every patch look at every other patch from the very first layer.

CNNs have built-in assumptions about images (nearby pixels belong together), which helps them learn from less data. ViT has fewer built-in assumptions, so it usually needs **a lot of training data**, or extra training tricks, to do well. When pre-trained on very large datasets, ViT matches or beats strong CNNs.

### Common mistakes

- **"ViT reads pixels one by one."** No. It reads **patches**. A 224 × 224 image becomes just 196 tokens, not 50,176.
- **"The patches are processed in order, like words in a sentence."** No. There's no causal mask; every patch sees every other patch at once. The position embeddings only tell the model where each patch came from.
- **"The [CLS] token comes from the image."** No. It's a learned vector that's the same for every image. It *collects* information about the image through attention.
- **"Bigger patches are always better."** No. Bigger patches mean fewer tokens and faster processing, but less fine detail.
- **Changing the image size changes N.** A model trained on 224 × 224 images has 197 position embeddings. For a different size, the position embeddings must be adapted, usually by stretching (interpolating) them.

## 7. Remember this

- ViT treats an image like a sentence: **patches are the words**.
- Pipeline: **image → patches → flatten → project → add [CLS] → add positions → transformer → classify**.
- Number of patches: **N = (H/P) × (W/P)**. Sequence length: **N + 1**.
- The **[CLS]** token gathers information from all patches and is used for the final answer.
- **No causal mask**: every patch attends to every other patch.
- ViT usually needs **lots of training data** to beat CNNs.

### Quick check

1. An image is 64 × 64 pixels, and the patch size is 8. How many patches are there? How long is the sequence?
2. A color patch is 8 × 8 pixels. How many numbers are in its flattened vector?
3. Why does ViT need position embeddings?
4. Which row of the transformer output is used to classify the image?

Try answering before reading on.

### Answers

1. **N = (64/8) × (64/8) = 8 × 8 = 64 patches.** With the [CLS] token, the sequence has **65** tokens.
2. **8 × 8 × 3 = 192** numbers (three channels: red, green, blue).
3. **Attention ignores order.** Without position embeddings, the model couldn't tell which patch came from where; a shuffled image would look the same to it.
4. **The [CLS] row**, the first row.

## 8. Going further

### How this connects to the other guides

```
Text:   text  → tokens  → token embeddings  ┐
                                            ├→  + positions → X → transformer
Image:  image → patches → patch embeddings  ┘
```

From X onward, a ViT and a text transformer work the same way. The only change is how the input sequence is made. That's why the same [Attention Blocks](attention_blocks.md) apply to both.

### A shortcut used in real code

Cutting into patches and multiplying by W_E is usually done in one step with a **convolution** whose kernel size and stride both equal P. It gives exactly the same result as Steps 2–4: each kernel position covers exactly one patch, with no overlap.

### Learn more

- 📖 [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al.): the original ViT paper
- 📖 [Vision Transformer (ViT)](https://huggingface.co/docs/transformers/model_doc/vit) (Hugging Face docs): overview, the [CLS] token, position embeddings, and how to use a pretrained ViT
- 📖 [vit-pytorch README](https://github.com/lucidrains/vit-pytorch/blob/main/README.md) (lucidrains): a short, readable ViT implementation, with many later variants
