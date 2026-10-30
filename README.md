# Easy Basics Learn

Different topics, explained in the easiest way possible.

Each guide starts from zero and builds up one small step at a time. If something can be said simply, it is said simply.

## Guides

| Topic | Guide | In one sentence |
|---|---|---|
| Machine learning | [MLP Inference: XOR](mlp_xor_inference.md) | How a small trained neural network turns two inputs into an answer, step by step |
| Machine learning | [Activation Functions](activation_functions.md) | The small non-linear functions (ReLU, sigmoid, softmax and many more) that let stacked layers learn curved patterns |
| Machine learning | [The Chain Rule](chain_rule.md) | The simple math rule that lets a network work out how each weight affects the error |
| Machine learning | [Backpropagation](backpropagation.md) | How a neural network learns by working out which weights to blame and nudging them |
| Machine learning | [Mini Autograd](mini_autograd.md) | How a tiny program works out gradients automatically and uses them to train a network on XOR |
| Machine learning | [Loss Functions](loss_functions.md) | How a model measures how wrong its predictions are, and why the choice of measure changes what it learns |
| Machine learning | [Overfitting and the Train/Test Split](overfitting_train_test_split.md) | Why a model can look good on its training data and still fail on new data, and how to check |
| Machine learning | [Regularization](regularization.md) | Ways to stop a model memorizing its training data so it does better on data it hasn't seen |
| Machine learning | [Tokenization](tokenization.md) | How text is cut into small pieces and turned into numbers a computer can use |
| Machine learning | [Attention Blocks](attention_blocks.md) | How each word in a sentence decides which other words to pay attention to |
| Machine learning | [Attention Masks](attention_masks.md) | Which tokens each token is allowed to look at, and the different kinds of masks |
| Machine learning | [LLM Inference](llm_inference.md) | How a language model writes its answer one token at a time |
| Machine learning | [Zero-Shot and Few-Shot Learning](zero_shot_few_shot_learning.md) | How a language model can do a new task just from a description or a few examples in the prompt, without retraining |
| Machine learning | [Chain-of-Thought Reasoning](chain_of_thought_reasoning.md) | Why a language model gets harder problems right when it writes out its steps before the answer |
| Machine learning | [In-Context Learning](in_context_learning.md) | How a language model picks up a new task from the text in its prompt, without any training |
| Machine learning | [RAG (Retrieval-Augmented Generation)](rag.md) | How a model looks up relevant documents first, then answers using them |
| Machine learning | [Tool Use and Function Calling](tool_use.md) | How a model asks programs to fetch information or take actions, and uses the results |
| Machine learning | [Agents](agents.md) | How a model reaches a goal by repeating think, act and observe in a loop |
| Machine learning | [Multimodality](multimodality.md) | How one model handles text, images and audio by turning them all into vectors |
| Machine learning | [Vector Databases](vector_databases.md) | How embeddings are stored and the most similar ones found quickly |
| Machine learning | [Quantization](quantization.md) | How storing weights with fewer bits makes models smaller and faster |
| Machine learning | [Distillation](distillation.md) | How a big teacher model trains a small student model to imitate it |
| Machine learning | [LoRA (Low-Rank Adaptation)](lora.md) | How to fine-tune a big model cheaply by training only a small add-on |
| Machine learning | [KV Cache](kv_cache.md) | How saving keys and values speeds up text generation, and what it costs in memory |
| Machine learning | [Mixture of Experts](mixture_of_experts.md) | How a router sends each token to a few expert networks out of many |
| Machine learning | [Vision Transformer (ViT)](vit.md) | How a transformer understands images by cutting them into patches and treating them like words |
| Information theory | [Shannon Entropy](shannon_entropy.md) | How much surprise or uncertainty a random outcome carries, measured in bits |
| Information theory | [Conditional Entropy](conditional_entropy.md) | The uncertainty left about one variable after you learn another |
| Information theory | [Mutual Information](mutual_information.md) | How much knowing one variable tells you about another |
| Information theory | [Information Gain](information_gain.md) | How much a question reduces uncertainty, used to choose decision tree splits |
| Information theory | [Cross-Entropy](cross_entropy.md) | How well predicted probabilities match real outcomes, measured as average surprise in bits |
| Information theory | [KL Divergence](kl_divergence.md) | The extra bits you pay for using a wrong probability distribution |
| Probability | [Bayesian Inference](bayesian_inference.md) | How to update a belief when new evidence arrives |
| Probability | [Expected Value and Variance](expected_value_variance.md) | Where a random number lands on average, and how far it tends to stray from that average |

## Learning path: information theory

Read these guides in order. Each one uses ideas from the one before it.

1. [Shannon Entropy](shannon_entropy.md): how much uncertainty is in one random outcome
2. [Conditional Entropy](conditional_entropy.md): how much uncertainty is left after you learn a clue
3. [Mutual Information](mutual_information.md): how much uncertainty the clue removed
4. [Information Gain](information_gain.md): using that number to pick the best question
5. [Cross-Entropy](cross_entropy.md): how surprised you are when your predicted probabilities are wrong
6. [KL Divergence](kl_divergence.md): just the extra surprise caused by the wrong predictions
7. [Bayesian Inference](bayesian_inference.md): how to update a belief when evidence arrives

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

The information theory guides also start with a **Learning path** line. Where the logarithms get tricky, they include a **Numbers you'll need** table.

## How to use these guides

- Start with **In simple words**. If you only read one section, read that one.
- Work through the example with pen and paper. Doing it yourself is how it sticks.
- Come back to **Remember this** for quick review.

## Adding a new guide

1. Create a new `.md` file named after the topic, for example `gradient_descent.md`. Use lowercase words joined by underscores (`snake_case`): no hyphens, spaces or capitals.
2. Follow the structure and the writing rules above.
3. Ask: "Could a beginner follow this?" If not, simplify.
4. Add a row to the **Guides** table, with a one-sentence explanation.
