# Agents: A Model That Works in a Loop

> **The one idea in this guide:** an **agent** is a language model that works toward a goal by repeating a loop: **think → act (use a tool) → look at the result → think again**, until the job is done.
>
> Everything below explains, demonstrates or deepens that idea.

## 1. In simple words

A normal chat model answers once and stops. An **agent** keeps going. You give it a goal, and it decides the steps itself:

```
Goal: "Find the cheapest flight to Rome next Friday and save it to my notes."

think   → I need flight prices.
act     → search_flights(to="Rome", date="Friday")
observe → 3 flights: €120, €95, €140
think   → €95 is cheapest. Now save it.
act     → save_note("Cheapest flight to Rome, Friday: €95")
observe → saved
think   → Done.
answer  → "The cheapest is €95. I've saved it to your notes."
```

**Analogy: an assistant with a to-do list.** You don't tell a good assistant every click. You say "book me a table for Friday", and they look up restaurants, check availability, call one, and come back when it's done. If a restaurant is full, they try another. That ability to **decide the next step from what just happened** is what makes an agent.

> **Quick check 1.** What's the main difference between a single tool call and an agent?
>
> *Answers to all checks are in Section 7.*

## 2. What you need to know first

- **[Tool Use](tool_use.md)**: how a model asks programs to do things. Agents are built on this. **Read it first.**
- **[Chain-of-Thought Reasoning](chain_of_thought_reasoning.md)**: agents often reason in steps before acting.
- **[LLM Inference](llm_inference.md)**: the model itself still just predicts tokens.

## 3. The key terms

| Term | Meaning |
|---|---|
| Agent | a model that chooses and takes steps in a loop to reach a goal |
| Goal | what the user wants done |
| Action | one step, usually a tool call |
| Observation | the result of an action, fed back to the model |
| Agent loop | the repeat of think → act → observe |
| Stopping condition | the rule for when the loop ends |
| Memory | what the agent keeps track of: the conversation so far, notes or files |
| Human in the loop | a person who approves or corrects important steps |

## 4. How it works

### The agent loop

```
              ┌──────────────────────────────────────┐
              ▼                                      │
   think: what's the next step?                      │
              │                                      │
   goal reached? ── yes ──▶ give final answer, stop  │
              │                                      │
              no                                     │
              ▼                                      │
   act: call a tool                                  │
              │                                      │
   observe: read the result ─────────────────────────┘
```

This loop is often called **ReAct** (**Re**ason + **Act**), after a 2022 paper that showed interleaving reasoning and actions works better than doing either alone.

### The parts of an agent

| Part | Job | Example |
|---|---|---|
| **Model** | the "brain": decides the next step | an LLM |
| **Tools** | the "hands": do things in the world | search, run code, read files |
| **Memory** | keeps track of progress | the conversation history, a notes file |
| **Instructions** | the rules it follows | "Ask before deleting anything." |
| **Loop code** | runs the cycle and enforces limits | a program that calls the model, runs tools, stops after 20 steps |

### Workflow vs agent

Not every multi-step system needs to be an agent.

| | Workflow | Agent |
|---|---|---|
| Who decides the steps? | **you**, in code, in advance | **the model**, while running |
| Example | always: search → summarise → email | "figure out what's needed and do it" |
| Predictable? | very | less |
| Good for | well-understood, repeated tasks | open-ended tasks where steps can't be known ahead |

**Rule of thumb:** use the simplest thing that works. Start with a single call, then a fixed workflow, and use a full agent only when the steps really can't be planned in advance.

## 5. Try it yourself

### Trace an agent by hand

**Goal:** "Which of my three files is the largest, and what's its first line?"

**Tools:** `list_files()`, `file_size(name)`, `read_first_line(name)`

| Step | Think | Act | Observe |
|---|---|---|---|
| 1 | I need the file names. | `list_files()` | `a.txt, b.txt, c.txt` |
| 2 | I need each size. | `file_size("a.txt")` | 120 bytes |
| 3 | | `file_size("b.txt")` | 940 bytes |
| 4 | | `file_size("c.txt")` | 310 bytes |
| 5 | b.txt is largest (940). Read it. | `read_first_line("b.txt")` | "Meeting notes, March" |
| 6 | I have everything. | (none) | |

**Final answer:** "b.txt is the largest (940 bytes). Its first line is 'Meeting notes, March'."

Notice that step 5 **depended on** what steps 2 to 4 returned. Nobody wrote that plan in advance; the agent chose it.

### Why long tasks are hard: errors add up

Suppose each step succeeds with probability p, and the task needs all n steps to succeed. Then the chance the whole task succeeds is:

```
P(success) = p × p × ... × p = pⁿ
```

| Success per step | 10 steps | 20 steps |
|---|---|---|
| 99% | 0.99¹⁰ ≈ **90%** | 0.99²⁰ ≈ **82%** |
| 95% | 0.95¹⁰ ≈ **60%** | 0.95²⁰ ≈ **36%** |
| 90% | 0.90¹⁰ ≈ **35%** | 0.90²⁰ ≈ **12%** |

A step that's "95% reliable" sounds good, but over 20 steps the agent finishes correctly only about a third of the time. This is why good agents **check their own work** and **recover from errors**: noticing a failed step and retrying it breaks this simple multiplication.

> **Quick check 2.** Each step succeeds 90% of the time. What's the chance 2 steps both succeed?

## 6. Variations, tips and mistakes

### Kinds of agents

| Kind | What it does |
|---|---|
| **Coding agent** | reads code, edits files, runs tests, fixes failures |
| **Research agent** | searches, reads pages, searches again, writes a report |
| **Computer-use agent** | looks at screenshots and clicks and types in apps |
| **Multi-agent system** | several agents with different roles (e.g. a planner and workers) |

### Ways to make agents more reliable

1. **Good tools** with clear descriptions and helpful error messages (see [Tool Use](tool_use.md)).
2. **Plan first:** ask the agent to write a short plan before acting.
3. **Check results:** run tests, re-read the file, compare with the goal.
4. **Limits:** a maximum number of steps, time or cost, so a confused agent can't loop forever.
5. **Human approval** before risky or irreversible actions (spending money, deleting, sending).
6. **Keep memory tidy:** long histories fill the context window; summarise or save notes to a file.

### Common mistakes

| Mistake | What goes wrong | Fix |
|---|---|---|
| Using an agent when a workflow would do | slower, costlier, less predictable | use fixed steps when you know them |
| No step limit | the agent loops forever or runs up a big bill | cap steps, time and cost |
| Too many permissions | a mistake (or a trick) causes real damage | give only the tools and access needed |
| Trusting everything it reads | a web page or email may contain instructions (**prompt injection**) | treat tool results as data, require approval for actions |
| No checking | early errors carry through the whole task | verify after important steps |
| Vague goal | the agent guesses what "done" means | state the goal and what success looks like |

## 7. Remember this

### Five essential ideas

1. **An agent is a loop:** think → act → observe → repeat, until done.
2. **The model chooses the steps**, based on what happened so far. That's the difference from a fixed workflow.
3. **Tools are its hands**; the model never acts directly, the loop code runs the tools.
4. **Errors compound:** pⁿ shrinks fast, so checking and recovery matter.
5. **Limit and supervise:** step limits, minimal permissions and human approval for risky actions.

### Exercises

1. An agent's steps each succeed 99% of the time. Roughly what's the chance a 10-step task fully succeeds?
2. For "every Monday, email me a summary of last week's sales", would you build a workflow or an agent? Why?
3. Name two actions an agent should ask a human about before doing.
4. In the file example, why couldn't the agent decide which file to read in step 1?

Try the exercises and the quick checks before reading on.

---

### Check your answers

**Quick checks**

1. An agent **repeats**: it decides the next step from each result, many times, until the goal is reached. A single tool call happens once.
2. 0.9 × 0.9 = **0.81 (81%)**.

**Exercises**

1. 0.99¹⁰ ≈ **90%**.
2. A **workflow**: the steps are always the same (get sales, summarise, email), so fixed code is simpler and more predictable.
3. For example: sending an email, deleting files, making a payment, posting publicly.
4. It didn't know the sizes yet. The choice depended on observations from later steps.

## 8. Going further

### Where this connects

- **[Tool Use](tool_use.md)**: the building block of every agent step.
- **[RAG](rag.md)**: a research agent can retrieve, read and retrieve again.
- **[Chain-of-Thought Reasoning](chain_of_thought_reasoning.md)**: the "think" part of the loop.
- **[Multimodality](multimodality.md)**: computer-use agents read screenshots.

### Short glossary

| Term | Meaning |
|---|---|
| Agent | a model acting in a loop toward a goal |
| ReAct | reason + act: interleaving thinking and tool calls |
| Observation | a tool's result, fed back to the model |
| Workflow | a fixed sequence of steps written in code |
| Stopping condition | when the loop ends |
| Human in the loop | a person who approves or corrects steps |
| Compounding errors | small per-step failure rates multiplying over many steps |

### Learn more

- 📖 [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) (Yao et al., 2022): the think-act-observe loop
- 📖 [Toolformer](https://arxiv.org/abs/2302.04761) (Schick et al., 2023): teaching models when to call tools
