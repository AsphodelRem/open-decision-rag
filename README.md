# DecisionRAG

**Let models navigate your documents.**

DecisionRAG organizes documents into a tree, uses model decisions to explore
relevant branches, and returns passages you can use to generate an answer.
Bring your own model endpoint, inspect retrieval decisions, and cache requests
for repeatable experiments.

This is my independent implementation of
[Retriever-Free Retrieval: Navigating Documents with Decision Models](https://arxiv.org/abs/2610.35868).
I am not an author of the paper. The project is a prototype with partial coverage
of the method; see [implementation status](docs/implementation_status.md).

## Why try it?

- **Explore hierarchical retrieval:** start with a flat corpus, then experiment
  with metadata or embedding-based document clusters.
- **Use your own models:** connect a chat model, a System One decision server,
  FRIDA-Decisions, or the supported direct DiffusionGemma backend on vLLM.
- **Inspect and repeat:** keep source IDs, numerical decision traces and model
  responses, with SQLite persistence and cache-only replay.

## Install

Requires Python 3.10+. From the repository root:

```sh
pip install .
```

## Ask your documents

This example uses a real chat model for relevance judging and answer generation.
Start an OpenAI-compatible server that supports chat completions and
`response_format={"type": "json_object"}`, then configure its URL and served
model name. Replace the values below with your deployment settings:

```sh
export MODEL_BASE_URL=http://localhost:8000/v1
export MODEL_NAME=your-served-chat-model
```

If authentication is required, also set `MODEL_API_KEY`.
Save the following as `quickstart.py` and run `python quickstart.py`:

```python
import asyncio
import os
import sys

from decisionrag import (
    CachedAsyncOpenAI,
    CorpusHierarchy,
    DecisionSearchEngine,
    Document,
    LLMOpenAI,
    SQLiteCache,
)


async def main() -> None:
    """Retrieve evidence and generate an answer using a real model endpoint."""
    async with CachedAsyncOpenAI(
        os.environ["MODEL_BASE_URL"],
        api_key=os.environ.get("MODEL_API_KEY"),
        cache=SQLiteCache("model-cache.sqlite3"),
    ) as client:
        model = LLMOpenAI(client, model_name=os.environ["MODEL_NAME"])
        corpus = await CorpusHierarchy().build_from_docs([
            Document(
                id="returns",
                title="Returns policy",
                text="Unused items can be returned within 30 days of delivery.",
            ),
            Document(
                id="shipping",
                title="Shipping",
                text="Standard delivery takes 3 to 5 business days.",
            ),
        ])
        engine = DecisionSearchEngine(corpus, decision_model=model, llm=model)
        answer = await engine.query("How long do I have to return an unused item?")
        sys.stdout.writelines((answer.response, "\n\n", answer.retrieval.to_text(), "\n"))


asyncio.run(main())
```

The flat corpus needs no embedding endpoint. Use `engine.search(...)` when you
only need passages. This chat adapter generates relevance estimates; native
decision-model readouts use separate adapters.

## Contribute

Contributions are welcome, especially model integrations, reproducible retrieval
experiments, bug fixes and clearer examples. Check the
[implementation status](docs/implementation_status.md) for open work. For a larger
change, open an issue first so we can agree on the scope.

To get started, fork the repository, create a branch and install development
dependencies:

```sh
pip install -e '.[dev]'
python -m flake8 --select=WPS,E999 --format=default odr tests examples
python -m mypy --config-file mypy.ini odr tests examples
python -m unittest discover -s tests -v
```

Keep changes focused, add regression tests for changed behavior, and include
type annotations and Sphinx-style docstrings. Preserve the public API outside
your change; fix lint findings without adding `noqa` or changing lint settings.
In your pull request, explain the problem, the resulting behavior and the checks
you ran. For model integrations, distinguish fixture tests from runs with real
weights and include deployment details when reporting results.
