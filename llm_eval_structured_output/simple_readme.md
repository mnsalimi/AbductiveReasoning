# Quick Start Guide

## 1. Setup Environment
Create a `.env` file in the root directory and add your API keys and base URLs. Depending on the model you intend to use, fill in the following:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=your_openai_base_url_here
GEMINI_BASE_URL=your_gemini_base_url_here
```

## 2. Setup Configuration
Open `config.py` to configure your evaluation settings:
- Set your preferred model in `JUDGE_MODEL` (e.g., `"gpt-4o-mini"` or `"gemini-2.0-flash"`).
- Adjust `N_SAMPLES` if needed.

> **⚠️ IMPORTANT: DO NOT MODIFY THE FOLLOWING VARIABLES IN `config.py`:**
> - `ACTIVE_METRICS`
> - `ACTIVE_DATASETS`
> - `EXCLUDED_CHECKPOINTS`

## 3. Run the Pipeline
Once your environment and basic configurations are set, you can run the main evaluation script:

```bash
python main.py
```
