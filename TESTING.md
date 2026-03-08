# Testing Guide

Step-by-step instructions for testing the KYC processor against each provider.

## 1. Environment Setup

```bash
cd /home/osboxes/projects-kyc-multi/kyc_processor-claude2

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: install native Google Gemini SDK
pip install "google-genai>=1.0"

# Copy environment template and add your API keys
cp .env.example .env
```

Edit `.env` and add API keys for the providers you want to test. You only need keys for the providers you plan to use.

## 2. Verify Provider Configuration

```bash
# Show all providers and whether they have keys configured
python -m src.cli list-providers
```

Expected output (green = ready, red = no key):

```
Default provider: openrouter

Provider         Name                   Status       Default Model
--------------------------------------------------------------------------------
parasail         Parasail.io            ready        Qwen/Qwen2.5-VL-72B-Instruct
openrouter       OpenRouter.ai          ready        qwen/qwen2.5-vl-72b-instruct
openai           OpenAI                 no key       gpt-4o
google           Google Gemini          ready        gemini-2.0-flash
...
```

## 3. Test Connectivity

Test each configured provider before processing documents:

```bash
# Test default provider
python -m src.cli test-connection

# Test specific providers
python -m src.cli test-connection --provider parasail
python -m src.cli test-connection --provider openrouter
python -m src.cli test-connection --provider google
python -m src.cli test-connection --provider fireworks
python -m src.cli test-connection --provider together
python -m src.cli test-connection --provider openai
```

Each should print `OK: Connection successful` in green.

## 4. Prepare Test Documents

Place identity document images in the `documents/` directory:

```bash
cp /path/to/test-passport.jpg documents/
cp /path/to/test-drivers-license.png documents/
```

Supported formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.webp`

**Important:** Do NOT commit real identity documents to git. The `documents/` directory contains only `.gitkeep` in version control. See [Keeping Files Out of Git](#7-keeping-transient-files-out-of-git) below.

## 5. Process Documents

### Single document

```bash
# Default provider, output to stdout
python -m src.cli process documents/test-passport.jpg

# Specific provider, save to file
python -m src.cli process documents/test-passport.jpg --provider google -o outputs/google-result.json

# Compare across providers
python -m src.cli process documents/test-passport.jpg --provider parasail -o outputs/parasail.json
python -m src.cli process documents/test-passport.jpg --provider openrouter -o outputs/openrouter.json
python -m src.cli process documents/test-passport.jpg --provider google -o outputs/google.json
python -m src.cli process documents/test-passport.jpg --provider fireworks -o outputs/fireworks.json
```

### Batch processing

```bash
# All documents with default provider
python -m src.cli batch

# Specific provider with custom output directory
python -m src.cli batch --provider google --output-dir outputs/google-batch/
```

Results are saved to:
- `outputs/individual/<filename>_result.json` — per-document extraction
- `outputs/batch_summary.json` — aggregate statistics

## 6. Run Unit Tests and Linter

```bash
# Unit tests
pytest tests/ -v

# Lint check
ruff check .

# Format check
ruff format --check .

# Auto-fix lint and format issues
ruff check --fix .
ruff format .
```

## 7. Keeping Transient Files Out of Git

The `.gitignore` already excludes transient files. Here's what stays out and why:

### Never committed (git-ignored):

| Pattern | Reason |
|---------|--------|
| `.env` | Contains API keys — **secrets** |
| `outputs/` | Generated results — regenerated each run |
| `.venv/` / `venv/` | Virtual environment — machine-specific |
| `__pycache__/` | Python bytecode |
| `*.log` | Log files |
| `.pytest_cache/` | Test cache |
| `.ruff_cache/` | Linter cache |

### Always committed:

| File | Reason |
|------|--------|
| `.env.example` | Template (no real keys) |
| `documents/.gitkeep` | Preserves empty directory |
| `providers.yaml` | Config (no secrets) |
| All `.py` files | Source code |

### Pre-commit safety check:

```bash
# Before any commit, verify no secrets or outputs are staged
git status

# Confirm .gitignore is working
git status --ignored

# Double-check no .env is staged
git diff --cached --name-only | grep -E '\.env$' && echo "WARNING: .env is staged!" || echo "OK"
```

### If you accidentally stage a secret:

```bash
# Unstage the file
git reset HEAD .env

# If already committed, remove from history
git rm --cached .env
git commit -m "Remove .env from tracking"
```

## 8. Provider-Specific Notes

### Parasail
- Free tier, no billing required
- Only supports Qwen models
- Does not support `response_format: json_object`

### OpenRouter
- Requires billing setup for paid models
- Free models available (check openrouter.ai/models)
- Supports automatic provider fallback

### Google Gemini
- If `google-genai` is installed: uses native SDK with `response_mime_type="application/json"` (more reliable structured output)
- If not installed: falls back to OpenAI-compatible endpoint (still works, but no native structured output enforcement)
- Test both modes:
  ```bash
  # With native SDK
  pip install "google-genai>=1.0"
  python -m src.cli test-connection --provider google
  # Shows: "Google Gemini (native SDK)"

  # Without native SDK
  pip uninstall google-genai
  python -m src.cli test-connection --provider google
  # Shows: "Google Gemini"
  ```

### Anthropic
- Not directly supported via OpenAI-compatible endpoint
- Use via OpenRouter: `--provider openrouter` with model `anthropic/claude-sonnet-4-20250514`

### Fireworks / Together
- Both are OpenAI-compatible, should work out of the box
- Fireworks is generally fastest for inference
