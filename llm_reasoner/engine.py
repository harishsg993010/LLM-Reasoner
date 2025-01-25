"""Core reasoning chain implementation."""

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Optional, Any, Union, cast

try:
    from litellm import acompletion
    from litellm.utils import ModelResponse
except ImportError:
    raise ImportError(
        "litellm is required for ReasonChain. "
        "Install it with `pip install litellm>=1.0.0`"
    )

from .settings import (
    SYSTEM_PROMPT, 
    MIN_STEPS, 
    DEFAULT_MAX_TOKENS, 
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT
)
from .models import ModelConfig, model_registry

class ReasoningError(Exception):
    """Error during reasoning chain execution."""
    pass

@dataclass
class Step:
    """A single step in the reasoning chain."""
    number: int
    title: str
    content: str
    confidence: float
    thinking_time: float
    is_final: bool = False

    @staticmethod
    def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from text that might contain non-JSON content."""
        try:
            # Try direct JSON parsing first
            text = text.strip()
            if text.startswith('{') and text.endswith('}'):
                return json.loads(text)
            # Look for JSON in markdown code blocks
            json_blocks = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
            for block in json_blocks:
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    continue
            # Try to find any JSON-like structure
            matches = re.findall(r'\{[^{}]*\}', text)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass
        return None

    @classmethod
    def _parse_xml_like_format(cls, text: str) -> Dict[str, Any]:
        """Parse XML-like format into a dictionary."""
        # Extract title
        first_line = text.split('\n')[0].strip()
        title_match = re.match(r'^(?:Step \d+:?\s*)?(.+)$', first_line)
        title = title_match.group(1) if title_match else "Unknown Step"

        # Extract thinking sections
        thinking_pattern = r'<thinking>(.*?)</thinking>'
        thinking_matches = re.findall(thinking_pattern, text, re.DOTALL)
        thinking_content = '\n'.join(thinking_matches) if thinking_matches else ''

        # Clean up content and determine if it's a final answer
        content = re.sub(thinking_pattern, '', text, flags=re.DOTALL).strip()
        is_final = bool(re.search(r'(?i)final[_ ]?answer|conclusion', content))

        return {
            'title': title,
            'content': content,
            'confidence': 0.8,  # Default confidence for non-JSON responses
            'next_action': 'final_answer' if is_final else 'continue'
        }

    @classmethod
    def from_response(cls, number: int, response: Union[str, Dict[str, Any]], thinking_time: float) -> 'Step':
        """Create a step from an LLM response."""
        try:
            # Handle string responses
            if isinstance(response, str):
                response_data = cls._extract_json_from_text(response)
                if not response_data:
                    response_data = cls._parse_xml_like_format(response)
            else:
                response_data = response

            if not response_data:
                raise ValueError("Empty or invalid response format")

            # Extract required fields with fallbacks
            title = response_data.get('title', '')
            if not title and isinstance(response, str):
                first_line = response.split('\n')[0].strip()
                title_match = re.match(r'^(?:Step \d+:?\s*)?(.+)$', first_line)
                if title_match:
                    title = title_match.group(1)

            # Determine if this is a final answer
            next_action = response_data.get('next_action', '').lower()
            is_final = (next_action == 'final_answer' or 
                       bool(re.search(r'(?i)final[_ ]?answer|conclusion', 
                                    response_data.get('content', ''))))

            return cls(
                number=number,
                title=title or f"Step {number}",
                content=response_data.get('content', response) if isinstance(response, str) else response_data.get('content', ''),
                confidence=float(response_data.get('confidence', 0.8)),
                thinking_time=thinking_time,
                is_final=is_final
            )
        except (KeyError, ValueError) as e:
            raise ReasoningError(f"Invalid response format: {str(e)}") from e

@dataclass
class ReasonChain:
    """Main reasoning chain implementation."""
    model_config: ModelConfig = field(default_factory=model_registry.get_default_model)
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    chat_history: List[Dict[str, str]] = field(default_factory=list)
    timeout: float = DEFAULT_TIMEOUT
    min_steps: int = MIN_STEPS

    def __init__(
        self,
        model: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: float = DEFAULT_TIMEOUT,
        min_steps: int = MIN_STEPS
    ) -> None:
        """Initialize the reasoning chain."""
        self.model_config = (
            model_registry.get_model(model)
            if model
            else model_registry.get_default_model()
        )
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.min_steps = min_steps
        self.chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]

    async def generate(self, query: str) -> AsyncIterator[str]:
        """Generate reasoning steps for a query."""
        async for step in self.generate_with_metadata(query):
            yield step.content

    async def _make_completion_request(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Make a completion request with timeout."""
        try:
            response = await asyncio.wait_for(
                acompletion(
                    model=self.model_config.name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                ),
                timeout=self.timeout
            )

            # Handle LiteLLM's ModelResponse type
            if isinstance(response, ModelResponse):
                if not response.choices or not response.choices[0].message:
                    raise ReasoningError("No valid choices in API response")
                content = response.choices[0].message.content
                if content is None:
                    raise ReasoningError("No content in API response")
                return {'choices': [{'message': {'content': content}}]}

            raise ReasoningError(f"Unexpected response type: {type(response)}")

        except asyncio.TimeoutError:
            raise ReasoningError(f"Request timed out after {self.timeout} seconds")
        except Exception as e:
            raise ReasoningError(f"API request failed: {str(e)}")

    async def generate_with_metadata(self, query: str) -> AsyncIterator[Step]:
        """Generate reasoning steps with metadata for a query."""
        try:
            self.chat_history.append({"role": "user", "content": query})
            step_number = 1

            while True:
                start_time = time.time()
                response = await self._make_completion_request(self.chat_history)

                thinking_time = time.time() - start_time
                content = cast(str, response['choices'][0]['message']['content'])
                step = Step.from_response(step_number, content, thinking_time)

                # Add step to chat history for context
                self.chat_history.append({
                    "role": "assistant",
                    "content": f"Step {step_number}: {step.content}"
                })

                yield step

                if step.is_final and step_number >= self.min_steps:
                    break

                step_number += 1

        except Exception as e:
            raise ReasoningError(f"Error during reasoning: {str(e)}") from e

    def clear_history(self) -> None:
        """Clear chat history except for the system prompt."""
        self.chat_history = [self.chat_history[0]]  # Keep system prompt