from contextlib import contextmanager
from typing import Any

from rich.text import Text

from ..reporting_utils import (
    convert_tqdm_to_rich,
    display_configuration,  # noqa: F401
    display_header,  # noqa: F401
    display_result,  # noqa: F401
    get_console,
    suppress_opik_logs,
)

PANEL_WIDTH = 70
console = get_console()

@contextmanager
def display_evaluation(
    message: str = "First we will establish the baseline performance:", verbose: int = 1
) -> Any:
    """Context manager to display messages during an evaluation phase."""
    # Entry point
    if verbose >= 1:
        console.print(Text(f"> {message}"))

    # Create a simple object with a method to set the score
    class Reporter:
        def set_score(self, s: float) -> None:
            if verbose >= 1:
                console.print(
                    Text(f"\r  Baseline score was: {s:.4f}.\n", style="green")
                )

    # Use our log suppression context manager and yield the reporter
    with suppress_opik_logs():
        with convert_tqdm_to_rich("  Evaluation", verbose=verbose):
            try:
                yield Reporter()
            finally:
                pass


def display_optimization_start_message(verbose: int = 1) -> None:
    if verbose >= 1:
        console.print(Text("> Starting the optimization run"))
        console.print(Text(""))