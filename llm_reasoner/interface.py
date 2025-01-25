"""Streamlit interface for LLM-Reasoner."""

try:
    import streamlit as st
except ImportError:
    raise ImportError(
        "streamlit is required for LLM-Reasoner UI. "
        "Install it with `pip install streamlit>=1.0.0`"
    )

import os
import asyncio
from typing import AsyncIterator, Optional
from llm_reasoner import engine
from llm_reasoner import models

DEFAULT_MAX_TOKENS = 750
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TIMEOUT = 30
MIN_STEPS = 1

def _format_step(step: engine.Step) -> None:
    """Format and display a reasoning step."""
    if step.is_final:
        st.success("Final Answer")
    else:
        st.subheader(f"Step {step.number}: {step.title}")

    st.write(step.content)
    st.progress(step.confidence)
    st.caption(f"Confidence: {step.confidence:.2f} | Thinking time: {step.thinking_time:.2f}s")

async def _stream_reasoning(chain: engine.ReasonChain, query: str) -> None:
    """Stream reasoning steps and update UI."""
    placeholder = st.empty()
    with st.spinner("Generating reasoning chain..."):
        try:
            async for step in chain.generate_with_metadata(query):
                with placeholder.container():
                    _format_step(step)
        except Exception as e:
            st.error(f"Error during reasoning: {str(e)}")

def _initialize_session_state() -> None:
    """Initialize session state variables."""
    if 'chain' not in st.session_state:
        st.session_state.chain = None
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []

def render_custom_model_form() -> None:
    """Render the custom model registration form."""
    with st.expander("Register Custom Model"):
        with st.form("custom_model_form"):
            model_name = st.text_input("Model Name", help="Name of your custom model")
            provider = st.text_input("Provider", help="Provider of the model (e.g., azure, custom-provider)")
            context_window = st.number_input("Context Window", min_value=1, value=4096, 
                                         help="Maximum context window size (optional)")

            submitted = st.form_submit_button("Register Model")

            if submitted and model_name and provider:
                try:
                    models.model_registry.register_model(
                        name=model_name,
                        provider=provider,
                        context_window=int(context_window)
                    )
                    st.success(f"Successfully registered model: {model_name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error registering model: {str(e)}")

def render_ui() -> None:
    """Render the main UI components."""
    st.title("LLM-Reasoner")
    st.write("Advanced reasoning chains with multiple LLM providers")

    # Add custom model registration form
    render_custom_model_form()

    with st.form("reasoning_form"):
        query = st.text_area("Enter your question:")
        col1, col2 = st.columns(2)

        with col1:
            model = st.selectbox(
                "Select model:",
                options=[m.name for m in models.model_registry.list_models().values()]
            )
            max_tokens = st.slider("Max tokens per response:", 100, 1000, DEFAULT_MAX_TOKENS)
            min_steps = st.slider("Minimum reasoning steps:", 1, 10, MIN_STEPS)

        with col2:
            temperature = st.slider("Temperature:", 0.0, 1.0, DEFAULT_TEMPERATURE)
            timeout = st.slider("Timeout (seconds):", 5, 60, int(DEFAULT_TIMEOUT))

        submitted = st.form_submit_button("Generate Reasoning Chain")

    if submitted and query:
        chain = engine.ReasonChain(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            min_steps=min_steps
        )
        try:
            asyncio.run(_stream_reasoning(chain, query))
            if 'query_history' in st.session_state:
                st.session_state.query_history.append(query)
        except Exception as e:
            st.error(f"Error: {str(e)}")

def main() -> None:
    """Main entry point for the Streamlit app."""
    st.set_page_config(
        page_title="LLM-Reasoner",
        page_icon="🤔",
        layout="wide"
    )
    _initialize_session_state()
    render_ui()

if __name__ == '__main__':
    main()