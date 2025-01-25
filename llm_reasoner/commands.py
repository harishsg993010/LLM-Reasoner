"""Command-line interface for LLM-Reasoner."""

import sys
import os
from typing import Optional
import asyncio
import traceback
import subprocess

try:
    import click
except ImportError:
    raise ImportError(
        "click is required for LLM-Reasoner CLI. "
        "Install it with `pip install click>=8.0.0`"
    )

try:
    from rich.console import Console
    from rich.progress import Progress
    from rich.panel import Panel
    from rich.table import Table
    from rich import print as rprint
except ImportError:
    raise ImportError(
        "rich is required for LLM-Reasoner CLI. "
        "Install it with `pip install rich>=12.0.0`"
    )

from .engine import ReasonChain, ReasoningError
from .models import model_registry

console = Console()

def display_available_models() -> None:
    """Display available models in a formatted table."""
    table = Table(title="Available Models")
    table.add_column("Name")
    table.add_column("Provider")
    table.add_column("Context Window")
    table.add_column("Default")

    for name, config in model_registry.list_models().items():
        table.add_row(
            name,
            config.provider,
            str(config.context_window) if config.context_window else "Unknown",
            "✓" if config.default else ""
        )

    console.print(table)

@click.group()
def cli() -> None:
    """Advanced reasoning chains with multiple LLM providers"""
    pass

DEFAULT_MAX_TOKENS = 750
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TIMEOUT = 30.0
MIN_STEPS = 1

@cli.command()
@click.argument('query')
@click.option('--model', '-m', help='Model to use for reasoning')
@click.option('--max-tokens', default=DEFAULT_MAX_TOKENS, help='Maximum tokens per response')
@click.option('--temperature', default=DEFAULT_TEMPERATURE, help='Temperature for response generation')
@click.option('--timeout', default=DEFAULT_TIMEOUT, help='Timeout in seconds for API requests')
@click.option('--min-steps', default=MIN_STEPS, help='Minimum number of reasoning steps')
@click.option('--debug', is_flag=True, help='Enable debug output')
def reason(query: str, model: Optional[str], max_tokens: int, temperature: float,
          timeout: float, min_steps: int, debug: bool) -> int:
    """Generate reasoning chain for the given query"""
    async def run_reasoning() -> int:
        try:
            if debug:
                console.print("[yellow]Debug mode enabled[/yellow]")
                console.print(f"Model: {model or 'default'}")
                console.print(f"Max tokens: {max_tokens}")
                console.print(f"Temperature: {temperature}")
                console.print(f"Timeout: {timeout}s")
                console.print(f"Min steps: {min_steps}")

            chain = ReasonChain(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                min_steps=min_steps
            )

            with Progress() as progress:
                task = progress.add_task("[cyan]Thinking...", total=1.0)

                async for step in chain.generate_with_metadata(query):
                    progress.update(task, completed=min(step.number / 20, 1.0))

                    if step.is_final:
                        console.print("\n[bold green]Final Answer:[/bold green]")
                        console.print(Panel(step.content))
                        console.print(f"Confidence: {step.confidence:.2f}")
                    else:
                        console.print(f"\n[bold]Step {step.number}: {step.title}[/bold]")
                        console.print(step.content)
                        console.print(f"Confidence: {step.confidence:.2f}")
                        console.print(f"Thinking time: {step.thinking_time:.2f}s")

                    if debug:
                        console.print(f"[dim]Message count: {len(chain.chat_history)}[/dim]")

            return 0

        except ReasoningError as e:
            console.print(f"[bold red]Reasoning Error:[/bold red] {str(e)}")
            if debug:
                console.print("[yellow]Debug traceback:[/yellow]")
                console.print(traceback.format_exc())
            return 1
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation cancelled by user[/yellow]")
            return 1
        except Exception as e:
            console.print(f"[bold red]Unexpected Error:[/bold red] {str(e)}")
            if debug:
                console.print("[yellow]Debug traceback:[/yellow]")
                console.print(traceback.format_exc())
            return 1

    return asyncio.run(run_reasoning())

@cli.command()
@click.argument('name')
@click.argument('provider')
@click.option('--context-window', type=int, help='Maximum context window size')
def register_model(name: str, provider: str, context_window: Optional[int]) -> int:
    """Register a new custom model.

    Example: llm-reasoner register-model my-custom-model azure --context-window 16384
    """
    try:
        model_registry.register_model(name, provider, context_window)
        console.print(f"[green]Successfully registered model {name}[/green]")
        display_available_models()
        return 0
    except ValueError as e:
        console.print(f"[red]Error registering model: {str(e)}[/red]")
        return 1

@cli.command()
def models() -> None:
    """List available models and their configurations"""
    display_available_models()

@cli.command()
@click.argument('model_name')
def set_model(model_name: str) -> int:
    """Set the default model for reasoning"""
    try:
        model_registry.set_default_model(model_name)
        console.print(f"[green]Successfully set {model_name} as default model[/green]")
        return 0
    except ValueError as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        display_available_models()
        return 1

@cli.command()
@click.option('--port', default=8501, help='Port to run the UI on')
def ui(port: int) -> None:
    """Launch the Streamlit UI"""
    try:
        console.print(f"[green]Starting LLM-Reasoner UI on port {port}...[/green]")
        # Get the path to the interface module
        package_dir = os.path.dirname(os.path.dirname(__file__))
        sys.path.insert(0, package_dir)
        interface_path = os.path.join(os.path.dirname(__file__), "interface.py")

        # Use streamlit run command directly
        subprocess.run([
            "streamlit", "run", interface_path,
            "--server.port", str(port),
            "--server.address", "0.0.0.0",
            "--browser.gatherUsageStats", "false"
        ], check=True)
    except ImportError:
        console.print("[red]Error: Streamlit is required for the UI.[/red]")
        console.print("Install it with: pip install streamlit>=1.0.0")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error launching UI: {str(e)}[/red]")
    except Exception as e:
        console.print(f"[red]Error launching UI: {str(e)}[/red]")