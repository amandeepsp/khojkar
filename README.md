# khojkar

khojkar (pa: ਖੋਜਕਾਰ, ipa: /kʰoːd͡ʒ.kɑːɾ/) is a deep research agent.

## Installation

### Prerequisites
- Python 3.12
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and resolver

## Quick Start (PyPI)

The easiest way to run khojkar is directly using `uvx`, which will fetch it from PyPI:

```bash
# Run directly from PyPI using uvx
uvx khojkar research --topic "Your research topic" --output report.md
```

To set up credentials when using this method, create a `.env` file in your current working directory with your API keys. `uvx` will automatically load environment variables from a `.env` file.

## Setup Credentials

khojkar uses LLMs via the LiteLLM library, which requires API keys for the models you want to use.

1. Create a `.env` file in your current working directory:

   ```bash
   touch .env
   ```

2. Add your API keys to the `.env` file. You only need the key for the model you intend to use. The default is `gemini/gemini-2.0-flash`, which requires the `GEMINI_API_KEY`. For other providers like OpenAI or Anthropic, you would use `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` respectively. You also need to add credentials for the Google Programmable Search Engine: `SEARCH_ENGINE_ID` and `SEARCH_ENGINE_API_KEY`.

   ```dotenv
   # Required for the default model (gemini/gemini-2.0-flash)
   GEMINI_API_KEY=your_gemini_api_key

   # Required for Google Programmable Search Engine
   SEARCH_ENGINE_ID=your_search_engine_id
   SEARCH_ENGINE_API_KEY=your_search_engine_api_key

   # Optional: add other LLM API keys as needed
   # OPENAI_API_KEY=your_openai_api_key
   # ANTHROPIC_API_KEY=your_anthropic_api_key
   ```

   *Note: Refer to the LiteLLM documentation for the specific environment variable names required for different LLM providers.*

## Usage

Basic Usage:
```bash
khojkar research --topic "Your research topic" --output report.md
```

Advanced Options:
```bash
# Use a different model
khojkar research --topic "Your research topic" --model "openai/gpt-4o" --output report.md

# Limit research steps
khojkar research --topic "Your research topic" --max-steps 5 --output report.md

# Use multi-agent research mode
khojkar research --topic "Your research topic" --multi-agent --output report.md
```

## License

See [LICENSE](LICENSE) file for details.
