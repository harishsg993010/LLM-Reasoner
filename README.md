# 🤔 LLM-Reasoner: Transform any LLM into a methodical thinker!

Make any LLM think deeper like OpenAI o1 and deepseek R1!

## ✨ Features

- 🧠 **Step-by-Step Reasoning**: No more black-box answers! See exactly how your LLM thinks
- 🔄 **Real-time Progress**: Watch the reasoning unfold with smooth animations
- 🎯 **Multi-Provider Support**: Works with all providers supported by LiteLLM
- 🎮 **Sweet UI**: A slick Streamlit interface
- 🛠️ **Power-User CLI**: For when you want to get nerdy with it
- 📊 **Confidence Tracking**: Know how sure your LLM is about each step

## 🚀 Installation

```bash
pip install llm-reasoner
```

## 🔑 Configuration

Set up your API keys:
```bash
# Choose your preferred provider:
export OPENAI_API_KEY="sk-your-key"      # OpenAI
export ANTHROPIC_API_KEY="your-key"      # Claude
export VERTEX_PROJECT="your-project"      # Google
```

## 🎮 Quick Start

```python
from llm_reasoner import ReasonChain
import asyncio

async def main():
    chain = ReasonChain(
        model="gpt-4",                # Choose your model
        min_steps=3,                  # Minimum reasoning steps
        temperature=0.2,              # Control creativity
        timeout=30.0                  # Set your timeout
    )

    async for step in chain.generate_with_metadata("Why is the sky blue?"):
        print(f"\nStep {step.number}: {step.title}")
        print(f"Thinking Time: {step.thinking_time:.2f}s")
        print(f"Confidence: {step.confidence:.2f}")
        print(step.content)

asyncio.run(main())
```

## 🛠️ CLI Usage

```bash
# List available models
llm-reasoner models

# Generate a reasoning chain
llm-reasoner reason "How do planes fly?" --min-steps 5

# Launch the UI
llm-reasoner ui
```

## 🔧 Supported Models

Out of the box support for:
- OpenAI: GPT-4, GPT-3.5-Turbo
- Anthropic: Claude 2
- Google: Gemini Pro
- Azure OpenAI models
- Custom models via LiteLLM

## 🧑‍💻 Development

1. Clone the repository:
```bash
git clone https://github.com/harishsg993010/LLM-Reasoner.git
cd LLM-Reasoner
```

2. Install development dependencies:
```bash
pip install -e ".[dev]"
```

3. Run tests:
```bash
pytest tests/ -v
```

## 📜 License

MIT - See [LICENSE](LICENSE) for details.

---

Made with 🧠 by Harish Santhanalakshmi Ganesan