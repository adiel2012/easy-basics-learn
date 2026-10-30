# Quantization: Smaller Numbers, Smaller Models

> **The one idea in this guide:** a model's weights are normally stored as precise decimal numbers using 16 or 32 bits each. **Quantization** stores them as **small integers** (8 or 4 bits) plus a **scale** to convert back. The model gets 2 to 8 times smaller and often faster, at the cost of a small rounding error.
>
> Everything below explains, demonstrates or deepens that idea.

## 1. In simple words

A language model is mostly a huge list of numbers (weights). A 7-billion-parameter model has 7 billion of them. How much memory they take depends on **how precisely** each one is stored:

| Format | Bits per weight | Memory for 7 billion weights |
|---|---|---|
| FP32 (32-bit float) | 32 | 28 GB |
| FP16 / BF16 | 16 | 14 GB |
| INT8 | 8 | 7 GB |
| INT4 | 4 | 3.5 GB |

(1 byte = 8 bits; 7 billion × 2 bytes = 14 GB, and so on.)

**Quantization** means going down this table: storing each weight with fewer bits.

**Analogy: rounding prices.** A shop could price items at €2.4718, but rounding to €2.47 (or even €2.50) loses almost nothing that matters and is much easier to write down. Quantization rounds the weights. The trick is rounding cleverly, so the model's answers barely change.

> **Quick check 1.** How much memory do 7 billion weights take at 8 bits each?
>
> *Answers to all checks are in Section 7.*

## 2. What you need to know first

Nothing is required: this guide explains what it uses. For more depth:

- **[LLM Inference](llm_inference.md)**: where the weights are used, and why decoding is limited by memory.

## 3. The key terms

| Term | Meaning |
|---|---|
| Bit | a 0 or 1; n bits can store 2ⁿ different values |
| Precision | how many bits each number uses |
| Floating point (FP) | a number format with a decimal point that can "float": FP32, FP16, BF16 |
| Integer (INT) | a whole number: INT8 holds −128 to 127; INT4 holds −8 to 7 |
| Scale | the number that converts between integers and real values |
| Quantize | real value → integer: divide by the scale, then round |
| Dequantize | integer → approximate real value: multiply by the scale |
| Quantization error | the difference between the original and the dequantized value |
| Outlier | a weight or activation much larger than the others |

## 4. How it works

### The basic recipe (symmetric quantization)

For a group of weights:

```
1. scale = (largest absolute value) ÷ (largest integer allowed)
2. q     = round(w ÷ scale)          ← store this small integer
3. ŵ     = q × scale                 ← use this when computing
```

For INT8 the largest integer we'll use is 127. For INT4 it's 7.

The weights are stored as the integers **q**, plus **one scale** per group. Since a group may hold dozens of weights, the scale adds very little memory.

### Why it saves time, not just memory

During decoding, the model generates one token at a time, and for each token it must **read every weight from memory**. Reading is often the bottleneck, not arithmetic. Fewer bytes to read means each token can come out faster.

### Why groups help

One huge weight (an **outlier**) forces a big scale, which makes every small weight in its group round badly. So weights are split into **small groups** (for example, 32 to 128 weights each), each with its own scale. An outlier then only hurts its own group.

## 5. Try it yourself

### Four weights to INT8

Weights: **w = [0.5, −1.2, 0.03, 2.4]**

**Step 1: the scale.**

```
largest absolute value = 2.4
scale = 2.4 ÷ 127 ≈ 0.01890
```

**Step 2: quantize** (divide by the scale, round):

| w | w ÷ scale | q (rounded) |
|---|---|---|
| 0.5 | 26.46 | **26** |
| −1.2 | −63.50 | **−64** (exactly halfway; rounds to −64) |
| 0.03 | 1.59 | **2** |
| 2.4 | 127.00 | **127** |

**Step 3: dequantize** (multiply back):

| w | q | ŵ = q × 0.01890 | Error ŵ − w |
|---|---|---|---|
| 0.5 | 26 | 0.4913 | −0.0087 |
| −1.2 | −64 | −1.2094 | −0.0094 |
| 0.03 | 2 | 0.0378 | +0.0078 |
| 2.4 | 127 | 2.4000 | 0.0000 |

Every error is below 0.01. Each weight now takes **1 byte** instead of 2 or 4.

### The same weights in INT4

Now only the integers −8 to 7 are allowed, so use 7 as the maximum:

```
scale = 2.4 ÷ 7 ≈ 0.3429
```

| w | w ÷ scale | q | ŵ | Error |
|---|---|---|---|---|
| 0.5 | 1.46 | 1 | 0.343 | −0.157 |
| −1.2 | −3.50 | −4 (halfway) | −1.371 | −0.171 |
| 0.03 | 0.09 | 0 | 0.000 | −0.030 |
| 2.4 | 7.00 | 7 | 2.400 | 0.000 |

Half the memory again, but the errors are about **20 times bigger**, and the small weight 0.03 became exactly 0. That's the trade-off: fewer bits, more rounding.

> **Quick check 2.** With the INT8 scale above, what integer does the weight 1.0 become?

## 6. Variations, tips and mistakes

### When is it done?

| Method | When | Effort | Quality |
|---|---|---|---|
| **Post-training quantization (PTQ)** | after training, on the finished model | low: no retraining | very good at 8 bits; usually good at 4 bits with smart methods |
| **Quantization-aware training (QAT)** | during training, the model practises with rounded weights | high | best at very low bits |

Smart PTQ methods (for example GPTQ and AWQ) choose roundings that protect the most important weights, using a small sample of real inputs.

### What gets quantized?

- **Weights only:** the most common for running LLMs locally. Weights are stored small and expanded to 16-bit when used.
- **Weights and activations:** also quantize the values flowing between layers, so the arithmetic itself runs in low precision. Faster on suitable hardware, but activations have more outliers and are harder to quantize well.
- **The [KV cache](kv_cache.md):** storing keys and values in 8 bits or less saves a lot of memory for long contexts.

### Common file formats you may see

Local LLM tools often share quantized models in formats like **GGUF**, with names such as `Q4_K_M` or `Q8_0`. The number is roughly the bits per weight (4 or 8); the letters describe the exact method.

### Tips and mistakes

| Tip / mistake | Why |
|---|---|
| 8-bit is usually a safe choice | quality loss is typically very small |
| 4-bit is a popular sweet spot for local use | big savings, usually modest quality loss |
| Below 4 bits, check quality carefully | errors grow fast, especially in smaller models |
| Larger models tolerate quantization better | a 4-bit large model often beats a 16-bit small model of the same size in memory |
| Always test on **your** task | average benchmarks can hide drops on specific skills like maths or code |
| Quantization isn't training | it doesn't teach the model anything new; it only stores it more compactly |

## 7. Remember this

### Five essential ideas

1. **Fewer bits per weight.** 32 → 16 → 8 → 4 bits halves the memory each step.
2. **Integers plus a scale.** q = round(w ÷ scale), and ŵ = q × scale.
3. **Rounding error is the cost.** It grows as the bits shrink.
4. **Small groups handle outliers**, each with its own scale.
5. **Faster, not just smaller:** decoding reads every weight for every token, so fewer bytes means faster tokens.

### Exercises

1. A 70-billion-parameter model in 4 bits: how many GB of weights?
2. Quantize w = [1.0, −0.4, 0.25] to INT8 with symmetric quantization. What are the scale and the integers?
3. Why does one very large weight make the other weights in its group less accurate?
4. How many different values can 4 bits represent?

Try the exercises and the quick checks before reading on.

---

### Check your answers

**Quick checks**

1. 7 billion × 1 byte = **7 GB**.
2. 1.0 ÷ 0.01890 ≈ 52.9 → **53**.

**Exercises**

1. 70 billion × 0.5 bytes = **35 GB**.
2. scale = 1.0 ÷ 127 ≈ 0.007874. Integers: 1.0 → **127**; −0.4 → −50.8 → **−51**; 0.25 → 31.75 → **32**.
3. The scale is set by the largest value. A big scale means each integer step is large, so small weights round to just a few levels (or to 0).
4. 2⁴ = **16**.

## 8. Going further

### Where this connects

- **[LoRA](lora.md)**: QLoRA fine-tunes a 4-bit quantized model with small LoRA adapters.
- **[Distillation](distillation.md)**: another way to make models smaller: train a smaller model instead of shrinking the numbers.
- **[KV Cache](kv_cache.md)**: the cache can also be quantized.
- **[Mixture of Experts](mixture_of_experts.md)**: huge MoE models are often quantized to fit in memory.

### Short glossary

| Term | Meaning |
|---|---|
| Quantization | storing numbers with fewer bits |
| Scale | converts between integers and real values |
| INT8 / INT4 | 8-bit / 4-bit integers |
| FP16 / BF16 | 16-bit floating-point formats |
| PTQ | quantizing after training |
| QAT | training with quantization in the loop |
| Outlier | an unusually large value |
| GGUF | a common file format for quantized local models |

### Learn more

- 📖 [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) (Dettmers et al., 2022): handling outliers in 8-bit
- 📖 [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) (Frantar et al., 2022)
- 📖 [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023)
