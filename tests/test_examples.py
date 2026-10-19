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
                    reason="TabPFN refuses training sets above 1000 rows on CPU")
def test_compare_predictors_example_runs(monkeypatch):
    pytest.importorskip("tabpfn", reason="requires the 'examples' extra")
    # the example imports from example_models/, which `python script.py` would put on the path
    monkeypatch.syspath_prepend(str(EXAMPLES))
    runpy.run_path(str(EXAMPLES / "compare_predictors_example.py"), run_name="__main__")
