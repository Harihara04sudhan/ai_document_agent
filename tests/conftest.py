"""Shared pytest configuration.

Provide a dummy Gemini key when none is configured so the suite runs the
same everywhere regardless of local .env contents. All LLM calls in tests
are mocked and never hit a real API.
"""

import os

os.environ.setdefault("GEMINI_API_KEY", "test-key-not-real")
