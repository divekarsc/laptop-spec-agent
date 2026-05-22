# Setup guide

Step-by-step instructions for running **laptop-spec-agent** on a new machine.

## 1. Requirements

| Requirement | Notes |
|-------------|--------|
| **Python 3.12+** | Check with `python3 --version` |
| **Git** | To clone the repository |
| **Google Gemini API key** | Free tier available at [Google AI Studio](https://aistudio.google.com/apikey) |
| **Network** | Scrape, LLM, and DuckDuckGo search need outbound HTTPS |

Optional but recommended: **[uv](https://docs.astral.sh/uv/getting-started/installation/)** for fast dependency installs.

## 2. Get the code

```bash
git clone <repository-url>
cd laptop-spec-agent
```

If you received a zip archive, unzip it and `cd` into the project folder instead.

## 3. Install dependencies

### Option A — uv (recommended)

```bash
# Install uv if needed: https://docs.astral.sh/uv/getting-started/installation/
uv sync
```

This creates `.venv` and installs everything from `pyproject.toml` / `uv.lock`.

### Option B — pip

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
# Or install from pyproject manually:
pip install langgraph langchain-google-genai langchain-community playwright pydantic python-dotenv ddgs duckduckgo-search
```

## 4. Install Playwright browsers

Chromium is required for scraping product pages.

```bash
uv run playwright install chromium
```

With pip/venv activated:

```bash
playwright install chromium
```

## 5. Configure API key

Secrets are **not** in the repo. Create a local `.env` from the template:

```bash
cp .env.example .env
```

Edit `.env` and set your key:

```env
GOOGLE_API_KEY=your_key_here
```

You can use `GEMINI_API_KEY` instead of `GOOGLE_API_KEY` if you prefer.

Optional settings (defaults are fine for a first run):

```env
# GEMINI_MODEL=gemini-2.5-flash
# PLAYWRIGHT_TIMEOUT_MS=60000
# PLAYWRIGHT_WAIT_UNTIL=domcontentloaded
```

**Do not commit `.env`** — it is listed in `.gitignore`.

## 6. Verify the install

Run these from the project root.

**Check Python imports:**

```bash
uv run python -c "from graph import build_graph; build_graph(); print('OK')"
```

**Check API key is loaded:**

```bash
uv run python -c "
from dotenv import load_dotenv
import os
load_dotenv()
print('API key set:', bool(os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')))
"
```

**Check Playwright:**

```bash
uv run python -c "
import asyncio
from playwright.async_api import async_playwright
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=True)
        await b.close()
    print('Playwright OK')
asyncio.run(main())
"
```

If any step fails, see [Troubleshooting](#troubleshooting) below.

## 7. First run

Pick a laptop **product detail page** URL (must start with `http://` or `https://`):

```bash
uv run python graph.py "https://www.lenovo.com/us/en/p/laptops/thinkpad/thinkpadx/thinkpad-x1-carbon-gen-13/21nxcto1wwus1"
```

When prompted, enter a short use case (max **25 words**), for example:

```text
Computer science studies, light coding, long battery life, portable
```

Or pass it on the command line:

```bash
uv run python graph.py "https://..." --use-case "Gaming and streaming, high refresh display"
```

You should see:

- Progress lines on the console (scrape → parse → optional search → fit evaluation → done)
- **Laptop specs** JSON, then **use case fit** (recommended / not recommended / compromise / insufficient data)
- Detailed logs in `logs/system.log`

A warning about missing fields after retries is normal if the page or search did not expose every spec.

## 8. What collaborators need to know

- **No Anthropic key** — the project uses **Google Gemini** only.
- **First run may be slow** — Playwright download + page load + LLM call.
- **Retail sites vary** — some block bots or never finish loading; adjust `PLAYWRIGHT_*` env vars or try another URL.
- **Logs** — share `logs/system.log` when reporting bugs (redact API keys if pasted).

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Missing GOOGLE_API_KEY` | Create `.env` from `.env.example` and set the key |
| `playwright install` / executable missing | Run `uv run playwright install chromium` |
| Page timeout | Increase `PLAYWRIGHT_TIMEOUT_MS=90000` in `.env` |
| `ModuleNotFoundError: ddgs` | Run `uv sync` or `pip install ddgs` |
| Gemini 429 / quota | Wait or change `GEMINI_MODEL` |
| Empty or partial JSON | Try another product URL; see README error handling section |

More detail: [README.md](../README.md).
