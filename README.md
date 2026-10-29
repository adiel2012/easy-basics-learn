# Easy Basics Learn

Different topics, explained in the easiest way possible.

Each guide starts from zero and builds up one small step at a time. If something can be said simply, it is said simply.

## Guides

| Topic | Guide | In one sentence |
|---|---|---|
| Machine learning | [MLP Inference: XOR](mlp_xor_inference.md) | How a small trained neural network turns two inputs into an answer, step by step |
| Machine learning | [The Chain Rule](chain_rule.md) | The simple math rule that lets a network work out how each weight affects the error |
| Machine learning | [Backpropagation](backpropagation.md) | How a neural network learns by working out which weights to blame and nudging them |
| Machine learning | [Mini Autograd](mini_autograd.md) | How a tiny program works out gradients automatically and uses them to train a network on XOR |
| Machine learning | [Tokenization](tokenization.md) | How text is cut into small pieces and turned into numbers a computer can use |
| Machine learning | [Attention Blocks](attention_blocks.md) | How each word in a sentence decides which other words to pay attention to |
| Machine learning | [Attention Masks](attention_masks.md) | Which tokens each token is allowed to look at, and the different kinds of masks |
| Machine learning | [LLM Inference](llm_inference.md) | How a language model writes its answer one token at a time |
| Machine learning | [Vision Transformer (ViT)](vit.md) | How a transformer understands images by cutting them into patches and treating them like words |
| Information theory | [Shannon Entropy](shannon_entropy.md) | How much surprise or uncertainty a random outcome carries, measured in bits |
| Information theory | [Cross-Entropy](cross_entropy.md) | How well predicted probabilities match real outcomes, measured as average surprise in bits |
| Information theory | [KL Divergence](kl_divergence.md) | The extra bits you pay for using a wrong probability distribution |
| Information theory | [Information Gain](information_gain.md) | How much a question reduces uncertainty, used to choose decision tree splits |
| Information theory | [Mutual Information](mutual_information.md) | How much knowing one variable tells you about another |
| Information theory | [Conditional Entropy](conditional_entropy.md) | The uncertainty left about one variable after you learn another |
| Probability | [Bayesian Inference](bayesian_inference.md) | How to update a belief when new evidence arrives |

## Code examples

These files go with the Mini Autograd guide. Keep them in the same folder.

| File | What it does |
|---|---|
| [mini_autograd.py](mini_autograd.py) | A small autograd engine: scalar values, gradients, and backpropagation |
| [xor_train.py](xor_train.py) | Trains a tiny neural network on XOR using the engine above. Run it with `python xor_train.py` |

## Our rules for writing simply

1. **Plain words first.** Explain the idea in everyday language before using any technical term.
2. **Define every new word.** The first time a term appears, say what it means.
3. **One idea at a time.** Each step adds just one new thing.
4. **Use analogies.** Compare new ideas to things people already know.
5. **Tiny examples.** Use numbers small enough to check with pen and paper.
6. **Show, don't just tell.** Use diagrams, tables and step-by-step calculations.
7. **Short sentences.** If a sentence needs to be read twice, split it.

## How each guide is structured

1. **In simple words**: the idea explained as if to a friend, with an analogy
2. **What you need to know first**: the few basics required
3. **The key terms**: a small table of the words and symbols used
4. **How it works**: the main idea, step by step
5. **Try it yourself**: a small worked example
6. **Variations**: common extensions and mistakes to avoid
7. **Remember this**: the key points in a few lines
8. **Going further**: how it connects to bigger topics

## How to use these guides

- Start with **In simple words**. If you only read one section, read that one.
- Work through the example with pen and paper. Doing it yourself is how it sticks.
- Come back to **Remember this** for quick review.

## Adding a new guide

1. Create a new `.md` file named after the topic, for example `gradient_descent.md`.
2. Follow the structure and the writing rules above.
3. Ask: "Could a beginner follow this?" If not, simplify.
4. Add a row to the **Guides** table, with a one-sentence explanation.
