"""Comprehensive tests for LLM-Reasoner."""
import pytest
import asyncio
from llm_reasoner import ReasonChain
from llm_reasoner.engine import ReasoningError, Step
from llm_reasoner.models import model_registry

def test_reasonchain_initialization():
    """Test basic initialization of ReasonChain."""
    chain = ReasonChain()
    assert chain is not None
    assert chain.model_config == model_registry.get_default_model()

def test_model_registry():
    """Test model registry initialization."""
    models = model_registry.list_models()
    assert isinstance(models, dict)
    assert len(models) > 0

def test_step_creation():
    """Test step creation from response."""
    step = Step.from_response(
        number=1,
        response={
            "title": "Test Step",
            "content": "Test content",
            "confidence": 0.8
        },
        thinking_time=1.0
    )
    assert step.number == 1
    assert step.title == "Test Step"
    assert step.content == "Test content"
    assert step.confidence == 0.8
    assert step.thinking_time == 1.0

@pytest.mark.asyncio
async def test_basic_generation():
    """Test basic chain generation."""
    chain = ReasonChain()
    question = "What are the implications of Moore's Law in modern computing?"
    step_count = 0

    try:
        async for step in chain.generate_with_metadata(question):
            # Basic structure checks
            assert step.number > 0
            assert isinstance(step.title, str)
            assert len(step.title) > 0
            assert isinstance(step.content, str)
            assert len(step.content) > 0

            # Value range checks
            assert 0 <= step.confidence <= 1
            assert step.thinking_time > 0

            step_count += 1
            if step.is_final or step_count >= 3:  # Limit steps for testing
                break
    except ReasoningError as e:
        pytest.fail(f"ReasoningError occurred: {str(e)}")
    except Exception as e:
        pytest.fail(f"Unexpected error: {str(e)}")

    assert step_count > 0, "Should generate at least one step"

@pytest.mark.asyncio
async def test_model_selection():
    """Test model selection and configuration."""
    chain = ReasonChain(model="gpt-3.5-turbo")
    assert chain.model_config.name == "gpt-3.5-turbo"
    assert chain.model_config.provider == "openai"