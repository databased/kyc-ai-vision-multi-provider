# KYC Vision Multi-Provider Document Processor

![CI](https://github.com/databased/kyc-ai-vision-multi-provider/actions/workflows/ci.yml/badge.svg)

AI-powered identity document processor that extracts structured KYC (Know Your Customer) data from passport, driver license, and national ID images using vision-capable LLMs across seven interchangeable providers.

## Features

- **7 LLM providers** — OpenRouter, Parasail, OpenAI, Google Gemini, Anthropic (via OpenRouter), Fireworks AI, Together AI
- **Pluggable architecture** — ABC base class + YAML config; add a provider with zero code changes
- **Native Google SDK** — optional `google-genai` for structured output; auto-falls back to OpenAI-compat
- **Batch processing** with parallel execution via `ThreadPoolExecutor`
- **Structured output** — per-document JSON + batch summary
- **Retry logic** — configurable attempts and delay per processing config
- **CLI** — Click-based with `process`, `batch`, `list-providers`, `test-connection`

## Quick Start

```bash
# Clone and install
git clone https://github.com/databased/kyc-ai-vision-multi-provider.git
cd kyc-ai-vision-multi-provider
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env — add at least one provider API key

# Place a document image in documents/
cp /path/to/passport.jpg documents/

# Process
python -m src.cli process documents/passport.jpg

# Batch process all images
python -m src.cli batch
```

See [TESTING.md](TESTING.md) for detailed step-by-step testing instructions across all providers.

## CLI Commands

```bash
# Process a single document
python -m src.cli process <image_path> [--provider google] [-o result.json]

# Batch process all images in documents/
python -m src.cli batch [--provider openrouter] [--input-dir docs/] [--output-dir out/]

# List providers and configuration status
python -m src.cli list-providers

# Test connectivity to a provider
python -m src.cli test-connection [--provider parasail]
```

Override the default provider with `--provider <name>` or set `KYC_PROVIDER` in your environment.

## Supported Providers

| Provider | Default Model | Pricing (per 1M tokens) | Notes |
|----------|--------------|------------------------|-------|
| OpenRouter | Qwen 2.5-VL 72B | $0.10 / $0.42 | 300+ models, auto-fallback |
| Parasail | Qwen 2.5-VL 72B | Free tier | OpenAI-compatible |
| OpenAI | GPT-4o | $2.50 / $10.00 | High quality |
| Google | Gemini 2.0 Flash | $0.075 / $0.30 | Native SDK or OpenAI-compat |
| Fireworks | Llama 4 Maverick | $0.20 / $0.20 | Fast inference |
| Together | Qwen2-VL 72B | $0.18 / $0.18 | Strong Qwen support |
| Anthropic | Claude Sonnet 4 | $3.00 / $15.00 | Use via OpenRouter |

## Project Structure

```
├── src/
│   ├── cli.py               # Click CLI entrypoint
│   ├── config.py             # YAML-driven provider config manager
│   ├── loader.py             # Document discovery and validation
│   ├── models.py             # Pydantic data models
│   ├── processor.py          # Vision processing + batch orchestration
│   └── clients/
│       ├── base.py           # ABC: BaseProviderClient
│       ├── factory.py        # Config → client routing
│       ├── openai_compat.py  # Generic OpenAI-compatible client
│       └── google_native.py  # Optional native Gemini SDK client
├── providers.yaml            # Provider and model definitions
├── .env.example              # Environment variable template
├── requirements.txt          # Python dependencies
├── TESTING.md                # Step-by-step testing guide
├── documents/                # Input document images
├── outputs/                  # Processing results (git-ignored)
├── tests/                    # Unit tests
└── .github/workflows/        # CI pipeline
```

## Architecture

The processor uses an **Abstract Base Class** pattern combined with **YAML-driven configuration**:

1. `BaseProviderClient` (ABC) — enforces `test_connection()` and `extract_identity_data()` on every client
2. `ProviderConfigManager` (singleton) — loads `providers.yaml` + `.env`
3. `get_client()` factory — reads config and returns the correct client implementation
4. `VisionProcessor` — sends images, parses JSON, handles retries
5. `BatchProcessor` — parallel processing with `ThreadPoolExecutor`

Adding a new provider requires only a YAML entry. If the provider is OpenAI-compatible, no code changes are needed. For providers needing a native SDK, add a client class implementing `BaseProviderClient`.

## Development

```bash
pip install -r requirements.txt
pytest tests/ -v
ruff check .
ruff format --check .
```

## License

MIT — see [LICENSE](LICENSE).

## Author

**Greg Hamer** — [github.com/databased](https://github.com/databased)

Part of the KYC Vision series:
- [parasail-ai-vision-demo-image-kyc_processor](https://github.com/databased/parasail-ai-vision-demo-image-kyc_processor) — Parasail provider
- [fireworks-ai-vision-demo-image-kyc_processor](https://github.com/databased/fireworks-ai-vision-demo-image-kyc_processor) — Fireworks AI provider
- **kyc-ai-vision-multi-provider** — Multi-provider (this repo)
