# Contributing to AI Document Agent

Thanks for your interest! Contributions of all sizes are welcome — bug reports, docs fixes, new features.

## Development setup

```bash
git clone https://github.com/Harihara04sudhan/ai_document_agent.git
cd ai_document_agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add a GEMINI_API_KEY for manual testing (tests use mocks)
```

## Before you open a PR

1. Run the test suite: `pytest tests/ -v`
2. Lint: `ruff check .` (CI enforces this)
3. Keep PRs focused — one feature or fix per PR
4. Add or update tests for any behavior change

## Good places to start

- Items on the [README roadmap](README.md#-roadmap)
- Issues labeled `good first issue`
- Improving test coverage for `processors/` and `agents/`

## Reporting bugs

Open an issue with: what you did, what you expected, what happened, and your Python version. A minimal reproduction helps a lot.

## Code style

- Follow existing patterns in the codebase
- Type hints on public functions
- Docstrings for modules and classes
- No secrets in code — use `.env`
