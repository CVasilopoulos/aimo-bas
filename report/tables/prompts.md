**System prompt** (`solution/aimo_bas/engine.py:SYSTEM_PROMPT`; identical to the organisers' `baselines/prompts/solve.txt`: True)

```
You are a precise math problem solver. Solve the given problem step by step, then output your final answer on the last line in the exact format:
ANSWER: <your answer>
This result should be an integer value or NaN if the given problem does not have a solution.
```

**Forced-answer suffix, `</think>` models** (`THINK_FORCE`)

```

</think>

ANSWER: 
```

**Forced-answer suffix, harmony (gpt-oss)** (`HARMONY_FORCE`)

```
<|end|><|start|>assistant<|channel|>final<|message|>ANSWER: 
```

**Reasoning split markers**: `</think>` and `<|end|>`.
