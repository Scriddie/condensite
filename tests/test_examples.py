import re
import runpy
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).parents[1]
EXAMPLES = ROOT / "src" / "examples"


def test_readme_example_matches_file():
    """ensure the example in the README matches the file."""
    readme = (ROOT / "README.md").read_text()
    block = re.search(r"```python\n(.*?)```", readme, re.DOTALL).group(1)
    assert block in (EXAMPLES / "readme_example.py").read_text()


def test_readme_example_runs():
    runpy.run_path(str(EXAMPLES / "readme_example.py"), run_name="__main__")


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="TabICL is too slow on CPU for this example")
def test_compare_predictors_example_runs(monkeypatch):
    pytest.importorskip("tabicl", reason="requires the 'examples' extra")
    # the example imports from example_models/, which `python script.py` would put on the path
    monkeypatch.syspath_prepend(str(EXAMPLES))
    runpy.run_path(str(EXAMPLES / "compare_predictors_example.py"), run_name="__main__")


def test_pit_boosting_example_runs():
    runpy.run_path(str(EXAMPLES / "pit_boosting_example.py"), run_name="__main__")
