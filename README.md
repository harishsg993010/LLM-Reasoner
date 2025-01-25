# 🤔 LLM-Reasoner: Make any LLM to think deeper like OpenAI o1 and deepseek R1!

Make any LLM to think deeper like OpenAI o1 and deepseek R1!

## ✨ What's Cool About It?

- 🧠 **Step-by-Step Reasoning**: No more black-box answers! See exactly how your LLM thinks, similar to O1's methodical approach
- 🔄 **Real-time Progress**: Watch the reasoning unfold with smooth animations
- 🎯 **Multi-Provider Support**: Works with all provider supported by LiteLLM
- 🎮 **Sweet UI**: A slick Streamlit interface to play with
- 🛠️ **Power-User CLI**: For when you want to get nerdy with it
- 📊 **Confidence Tracking**: Know how sure your LLM is about each step

## 🚀 Quick Start

Pop this in your terminal:
```bash
pip install llm-reasoner
```

Got API keys? Drop 'em in:
```bash
# Pick your flavor:
export OPENAI_API_KEY="sk-your-key"      # OpenAI fan?
export ANTHROPIC_API_KEY="your-key"      # Team Claude?
export VERTEX_PROJECT="your-project"     # Google enthusiast?
```

## 🎮 Jump Right In!

```bash
# List your available models
llm-reasoner models

# Generate a reasoning chain
llm-reasoner reason "How do planes fly?" --min-steps 5

# Launch the UI
llm-reasoner ui
```

### Let's see it in action as SDK 

```python
from llm_reasoner import ReasonChain
import asyncio

async def main():
    # Create a chain with your preferred settings
    chain = ReasonChain(
        model="gpt-4",                # Choose your model
        min_steps=3,                  # Minimum reasoning steps
        temperature=0.2,              # Control creativity
        timeout=30.0                  # Set your timeout
    )

    # Watch it think step by step!
    async for step in chain.generate_with_metadata("Why is the sky blue?"):
        print(f"\nStep {step.number}: {step.title}")
        print(f"Thinking Time: {step.thinking_time:.2f}s")
        print(f"Confidence: {step.confidence:.2f}")
        print(step.content)

asyncio.run(main())
```

## 🌟 Cool Features You'll Love

### Rich Metadata for Each Step
```python
async for step in chain.generate_with_metadata(query):
    print(f"Title: {step.title}")         # What's this step about?
    print(f"Content: {step.content}")     # The actual thinking
    print(f"Confidence: {step.confidence}")  # How sure is it?
    print(f"Time: {step.thinking_time}s") # How long did it take?
```

### Custom Model Registration
Want to use your own models? We've got you covered! You can register custom models through both Python and CLI:

```python
from llm_reasoner import model_registry

# Add your own models
model_registry.register_model(
    name="my-cool-model",
    provider="custom-provider",
    context_window=8192
)
```

Using the CLI:
```bash
# Register a new model
llm-reasoner register-model my-custom-model azure --context-window 16384

# List all available models (including your custom ones)
llm-reasoner models

# Set your custom model as default
llm-reasoner set-model my-custom-model

# Use your custom model
llm-reasoner reason "What is quantum computing?" --model my-custom-model
```

### 4. Interactive UI Features
- Drag-n-drop model selection
- Real-time confidence visualization
- Step-by-step animations
- Custom parameter tuning
- Query history tracking

## 🎛️ Power User Settings

Fine-tune your chains:

```python
chain = ReasonChain(
    model="claude-2",            # Pick your model
    max_tokens=750,             # Control response length
    temperature=0.2,            # Adjust randomness
    timeout=30.0,               # Set API timeout
    min_steps=5                 # Minimum reasoning steps
)

# Clear history if needed
chain.clear_history()
```

## 🔧 Model Support

Out of the box, we support:
- OpenAI: GPT-4, GPT-3.5-Turbo
- Anthropic: Claude 2
- Google: Gemini Pro
- Azure OpenAI models
- Custom models through our flexible provider system LiteLLM

Need to use a different model? Just register it with our CLI or Python API!

## 🎨 UI Walkthrough

1. Launch with `llm-reasoner ui`
2. Pick your model from the dropdown
3. Adjust settings with sliders
4. Type your question
5. Watch the magic happen with real-time updates!

## 🤓 Pro Tips

- Use `min_steps` to ensure thorough reasoning
- Lower `temperature` for more consistent results
- Watch the confidence scores to gauge reliability
- Try different models for different types of questions

## 🔮 What's Next?

We're constantly adding cool new features! Got ideas? We'd love to hear them!

## 📜 License

MIT - Go wild! Just give us a shoutout if you make something awesome with it.

---

Made with 🧠 and ❤️ by the Harish Santhanalakshmi Ganesan Happy reasoning! 🚀
