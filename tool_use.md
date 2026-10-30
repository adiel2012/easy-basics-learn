# Tool Use and Function Calling: Letting a Model Ask for Help

> **The one idea in this guide:** a language model can only write text. With **tool use**, it writes a structured **request** ("call the weather tool for Paris"), your program **runs** the tool, and the **result** is given back to the model so it can finish the answer.
>
> Everything below explains, demonstrates or deepens that idea.

## 1. In simple words

An LLM is good at language, but on its own it:

- can't check today's weather, a stock price or your calendar,
- can be unreliable at exact arithmetic,
- can't send an email, save a file or book a meeting.

**Tool use** (also called **function calling**) gives it a way to ask other programs to do those things.

```
User:   What's the weather in Paris?
Model:  [calls tool]  get_weather(city="Paris")
App:    [runs it]     → {"temp_c": 18, "sky": "cloudy"}
Model:  It's 18 °C and cloudy in Paris.
```

**Analogy: a manager with a phone.** The manager (the model) can't do everything personally, but knows who to call. They phone accounting ("what's last month's total?"), wait for the answer, then write the report. The manager never touches the spreadsheet; they just ask the right person the right question.

> **Important:** the model **never runs the tool itself**. It only writes the request. Your program decides whether to run it.

> **Quick check 1.** Who actually runs the tool: the model or the application?
>
> *Answers to all checks are in Section 7.*

## 2. What you need to know first

Nothing is required: this guide explains what it uses. For more depth:

- **[LLM Inference](llm_inference.md)**: the model writes one token at a time, and a tool call is just tokens in a special format.
- **[In-Context Learning](in_context_learning.md)**: the tool descriptions sit in the context, so the model learns which tools exist from the prompt.

## 3. The key terms

| Term | Meaning |
|---|---|
| Tool / function | a program the model can ask to use: a search, a calculator, an API |
| Tool definition | a description of the tool: its name, what it does, and its inputs |
| Schema | the exact shape of the inputs, usually written in **JSON Schema** |
| Tool call | the model's structured request to use a tool, with arguments |
| Arguments | the input values in the tool call, like `city = "Paris"` |
| Tool result | what the tool returned, sent back to the model |
| JSON | a simple text format for structured data: `{"key": "value"}` |

## 4. How it works

### Step 0: describe the tools

Before the conversation, you tell the model which tools exist. Each definition has a **name**, a **description** and a **schema** for its inputs:

```json
{
  "name": "get_weather",
  "description": "Get the current weather for a city.",
  "input_schema": {
    "type": "object",
    "properties": {
      "city": { "type": "string", "description": "City name, e.g. Paris" }
    },
    "required": ["city"]
  }
}
```

The description matters a lot. The model reads it to decide **when** to use the tool and **what** to pass in.

### The tool-use loop

```
        ┌─────────────────────────────────────────────────┐
        ▼                                                 │
  model reads: tools + conversation                       │
        │                                                 │
  does it need a tool?                                    │
        │                                                 │
   yes ─┴─ no                                             │
    │       │                                             │
    │       └──▶ writes the final answer → done            │
    ▼                                                     │
  writes a tool call (name + arguments)                   │
        │                                                 │
  app checks it, runs the tool, gets a result             │
        │                                                 │
  app adds the result to the conversation ────────────────┘
```

1. **Decide.** The model reads the question and the tool descriptions, and decides whether a tool would help.
2. **Call.** It writes a tool call in a fixed format instead of a normal reply.
3. **Run.** Your application reads the call, **checks** it, and runs the real function.
4. **Return.** The application adds the result to the conversation.
5. **Continue.** The model reads the result and either answers or calls another tool.

### How does the model "know" the format?

Models are **trained** to produce tool calls in a specific structured format when tools are available. The tool definitions are placed in the context (often inside the system prompt), so from the model's point of view, deciding to call a tool is just predicting the right tokens.

## 5. Try it yourself

### A calculator tool

**Tool definition:**

```json
{
  "name": "multiply",
  "description": "Multiply two numbers exactly.",
  "input_schema": {
    "type": "object",
    "properties": { "a": {"type": "number"}, "b": {"type": "number"} },
    "required": ["a", "b"]
  }
}
```

**The conversation, turn by turn:**

| # | Who | What |
|---|---|---|
| 1 | User | "What is 4,821 × 37?" |
| 2 | Model | tool call: `multiply(a = 4821, b = 37)` |
| 3 | App | runs it: 4821 × 37 = **178,377** |
| 4 | App → model | tool result: `178377` |
| 5 | Model | "4,821 × 37 = 178,377." |

Check step 3 by hand: 4,821 × 37 = 4,821 × 30 + 4,821 × 7 = 144,630 + 33,747 = **178,377**. ✓

### Two tools in a row

**Question:** "Is it warmer in Paris or Rome right now?"

| # | Who | What |
|---|---|---|
| 1 | Model | `get_weather(city = "Paris")` |
| 2 | App | `{"temp_c": 18}` |
| 3 | Model | `get_weather(city = "Rome")` |
| 4 | App | `{"temp_c": 24}` |
| 5 | Model | "Rome is warmer: 24 °C vs 18 °C in Paris." |

Many systems let the model request **both calls at once** (parallel tool calls), because they don't depend on each other. That saves a round trip.

> **Quick check 2.** The user asks "What's the capital of France?" and a weather tool is available. Should the model call it?

## 6. Variations, tips and mistakes

### Common kinds of tools

| Kind | Examples |
|---|---|
| Information | web search, database lookup, [RAG](rag.md) search, weather |
| Calculation | calculator, running code |
| Actions | send email, create a calendar event, write a file |
| Other models | image generation, speech-to-text |

**Actions are riskier than information.** Reading the weather is harmless if it goes wrong. Sending an email or deleting a file is not. Many apps ask the user to approve actions first.

### Tips for writing tools

1. **Clear names and descriptions.** "get_weather: current weather for a city" beats "tool1".
2. **Say when to use it**, and when not to.
3. **Keep inputs simple**, with types and examples.
4. **Return useful errors.** "City not found: 'Pariss'. Did you mean 'Paris'?" lets the model fix its call.
5. **Fewer, well-designed tools** usually work better than many overlapping ones.

### Common mistakes

| Mistake | What goes wrong | Fix |
|---|---|---|
| Running tool calls without checking | a bad or harmful argument goes straight to a real system | validate arguments; ask the user before risky actions |
| Vague descriptions | the model picks the wrong tool, or none | describe purpose, inputs and when to use it |
| Huge tool results | the context fills up, and costs rise | return only what's needed |
| Trusting tool results as instructions | a web page or email may contain text like "ignore your instructions" (**prompt injection**) | treat results as data, not commands |
| Assuming the model is always right | it can pass wrong arguments or invent a tool name | check, and return clear errors |

### Standards for connecting tools

Instead of every app defining tools its own way, standards such as the **Model Context Protocol (MCP)** describe a common way for tools and data sources to plug into AI applications.

## 7. Remember this

### Five essential ideas

1. **The model writes requests, not actions.** It produces a structured tool call; the app runs it.
2. **Tools are described in the context:** name, description, input schema.
3. **It's a loop:** decide → call → run → return → continue, until the model answers.
4. **Tools fill the model's gaps:** fresh information, exact calculation, real-world actions.
5. **Check before you run.** Validate arguments, confirm risky actions, and treat results as data.

### Exercises

1. Write a tool definition (name, description, inputs) for a tool that converts a temperature from Celsius to Fahrenheit.
2. A user asks "Add 'dentist, Friday 3 pm' to my calendar." Which kind of tool is this, and why might the app ask for confirmation?
3. The model calls `get_weather(city = "Pariss")`. What should the tool return?
4. Why should the text a web-search tool returns not be treated as instructions?

Try the exercises and the quick checks before reading on.

---

### Check your answers

**Quick checks**

1. **The application.** The model only writes the request.
2. **No.** It already knows the answer (Paris), and weather doesn't help.

**Exercises**

1. For example: name `celsius_to_fahrenheit`; description "Convert a temperature from Celsius to Fahrenheit"; input `celsius` (number, required).
2. An **action** tool. It changes something real, and a wrong date or time would be a real mistake.
3. A clear error, for example `"City not found: 'Pariss'. Did you mean 'Paris'?"`, so the model can retry.
4. Web pages are written by anyone. They might contain text designed to trick the model (prompt injection).

## 8. Going further

### Where this connects

- **[Agents](agents.md)**: a model that uses tools in a loop to finish multi-step tasks.
- **[RAG](rag.md)**: retrieval can be a tool the model chooses to call.
- **[In-Context Learning](in_context_learning.md)**: tool definitions are context the model reads.

### Short glossary

| Term | Meaning |
|---|---|
| Tool use / function calling | the model asks a program to do something, and gets the result back |
| Tool definition | name, description and input schema |
| JSON Schema | a standard way to describe the shape of JSON data |
| Tool call | the model's structured request |
| Tool result | the output returned to the model |
| Parallel tool calls | several independent calls requested at once |
| MCP | Model Context Protocol: a standard for connecting tools to AI apps |

### Learn more

- 📖 [Toolformer](https://arxiv.org/abs/2302.04761) (Schick et al., 2023): a model that learns when to call tools
- 📖 [Model Context Protocol](https://modelcontextprotocol.io): the open standard for connecting tools and data to AI applications
