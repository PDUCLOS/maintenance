# Evaluation methodology

## Why RAGAS

RAGAS is the de facto standard for RAG evaluation as of 2026. It computes
LLM-judged metrics without requiring human-annotated ground truth beyond
the question/answer pairs. We use it both for **regression detection**
(does a code change degrade retrieval quality?) and for **tuning guidance**
(which prompts and chunking strategies move the needle?).

## The 4 metrics we track

| Metric | What it measures | Why it matters | Target |
|--------|------------------|----------------|--------|
| **Faithfulness** | Is the answer faithful to the retrieved context (no hallucination)? | Recruiter red flag #1: making up sensor values | > 0.75 |
| **Answer relevancy** | Is the answer relevant to the question? | Detects generic / off-topic answers | > 0.75 |
| **Context precision** | Are the retrieved chunks the right ones? | Measures retriever quality | > 0.70 |
| **Context recall** | Did we retrieve ALL the chunks we needed? | Catches thin retrievals | > 0.65 |

## Evaluation dataset

`data/processed/eval_dataset.jsonl` contains **30 Q&A pairs** generated
deterministically from the PDF catalogue (random seed = 42):

- **10 factual** — "How many engines are in FD001?", "What is the mean of
  sensor_11 across all cycles in FD002?"
- **10 reasoning** — "Does sensor_X tend to increase or decrease as the
  degradation?"
- **5 retrieval** — "Which document discusses X?" (filename + page
  answer; checks the retriever surfaces the right page)
- **5 out-of-scope** — "What is the phone number of NASA support?" (the
  answer should be "I don't know from the available data.")

The ground-truth answers are extracted from the actual PDF pages, not
written by hand, so they stay accurate as long as the catalogue
doesn't change.

## Workflow

```bash
# Generate the dataset (deterministic)
make eval-dataset

# Run RAGAS and snapshot the scores
make eval

# Inspect the latest snapshot
cat reports/eval_$(ls reports/ | grep eval_ | sort | tail -1 | sed 's/.json//').json
```

Each run writes a new file in `reports/`, never overwrites. That gives us
a time series of scores across tuning iterations.

## Baseline targets (PLAN §6 / §7)

| Metric | W3 baseline | W4 target |
|--------|-------------|-----------|
| Faithfulness | ~0.50 | > 0.75 |
| Answer relevancy | ~0.60 | > 0.75 |
| Context precision | ~0.50 | > 0.70 |
| Context recall | ~0.45 | > 0.65 |
| Latency (M-series CPU) | ~15 s | < 10 s |
| Latency (M-series GPU) | ~5 s | < 5 s |

## Tuning levers (W4)

When baseline < target:

1. **System prompt** — make it more directive ("answer only from context")
2. **Chunk size** — try 300 / 500 / 800 tokens
3. **Hybrid search** — add BM25 to dense retrieval (RRF)
4. **Reranking** — add `cross-encoder/ms-marco-MiniLM-L-6-v2` on top-k
5. **Embedding model** — try `bge-large` instead of `bge-small`
6. **LLM** — currently `Qwen2.5-7B-Instruct` (chosen over `Mistral-7B-Instruct-v0.3`
   after an A/B test on agent tool-calling reliability, see PLAN.md §8); if scores
   plateau, try `Qwen2.5-14B-Instruct` or `Llama-3-8B-Instruct`

Each change re-runs `make eval`. The snapshot diff is the tuning
evidence we put in the README and in the interview pitch.
