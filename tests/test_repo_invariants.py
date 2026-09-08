"""The architecture rules, as tests.

``_core-checklist.md`` states each critical pattern with a grep that verifies it. A grep
only runs when someone remembers to run it; these run on every commit. If one fails, the
rule it encodes has been broken — fix the code, not the test.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from startup_outcomes.config import EXCLUDED_LEAKY, PROJECT_ROOT

SOURCE_ROOT = PROJECT_ROOT / "src" / "startup_outcomes"
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks"

#: The three modules that legitimately enumerate the whole schema, leaky columns included.
SCHEMA_MODULES = {"config.py", "load.py", "audit.py"}


def source_files() -> list[Path]:
    return sorted(SOURCE_ROOT.rglob("*.py"))


def offending_lines(pattern: str, *, skip: set[str] | None = None) -> list[str]:
    """Every line in ``src/`` matching ``pattern``, as ``path:lineno: text``."""
    skipped = skip or set()
    found = []
    for path in source_files():
        if path.name in skipped:
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if re.search(pattern, line):
                found.append(f"{path.relative_to(PROJECT_ROOT)}:{number}: {line.strip()}")
    return found


def test_no_module_outside_the_schema_three_names_a_leaky_column() -> None:
    """The timing contract's enforcement.

    The leakage demonstration reaches these columns through ``config.EXCLUDED_LEAKY``, so
    it stays inside the rule rather than becoming an exception to it. If a module spells
    a name directly, this grep stops being meaningful and the contract quietly dies.
    """
    pattern = "|".join(re.escape(column) for column in EXCLUDED_LEAKY)
    assert offending_lines(pattern, skip=SCHEMA_MODULES) == []


def test_all_randomness_derives_from_the_one_seed() -> None:
    """Every line mentioning randomness must name ``RANDOM_SEED`` on the same line."""
    found = [
        line
        for line in offending_lines(r"random_state|np\.random|random_seed")
        if "RANDOM_SEED" not in line
    ]
    assert found == []


def test_nothing_iterates_over_rows() -> None:
    assert offending_lines(r"iterrows|itertuples|\.apply\(lambda row") == []


def test_nothing_in_src_writes_to_disk() -> None:
    """Persist-nothing, as a gate rather than a claim.

    A saved model or CSV is a second source of truth that drifts from the code that made
    it, and ``data/`` is git-ignored so review would never see it.
    """
    assert offending_lines(r"to_csv|to_parquet|to_pickle|joblib|savefig|\.write\(") == []


def test_nothing_in_src_suppresses_warnings() -> None:
    """``mlkit`` opens with ``filterwarnings("ignore")``. This project does the opposite."""
    assert offending_lines(r"filterwarnings|simplefilter") == []


def test_no_plotting_module_touches_global_state() -> None:
    """No pyplot import anywhere, so figures cannot depend on import or call order.

    Matches import statements rather than the word, so a docstring explaining the rule
    does not trip the rule.
    """
    assert offending_lines(r"^\s*(?:import|from)\s+.*pyplot|(?<![\w.])plt\.") == []


def test_nothing_imports_the_reference_workbooks() -> None:
    """``references/`` is unlinted third-party course material and must stay unimported."""
    importing = re.compile(r"^\s*(?:from|import)\s+mlkit\b")
    for root in (SOURCE_ROOT, PROJECT_ROOT / "tests"):
        for path in sorted(root.rglob("*.py")):
            for number, line in enumerate(path.read_text().splitlines(), start=1):
                assert not importing.match(line), f"{path}:{number}"


@pytest.mark.parametrize("notebook", sorted(NOTEBOOK_ROOT.glob("*.ipynb")))
def test_notebooks_define_no_reusable_logic(notebook: Path) -> None:
    """Anything worth keeping is worth linting and testing, so it belongs in ``src/``."""
    text = notebook.read_text()
    assert '"def ' not in text, f"{notebook.name} defines a function"
    assert '"class ' not in text, f"{notebook.name} defines a class"


@pytest.mark.parametrize("notebook", sorted(NOTEBOOK_ROOT.glob("*.ipynb")))
def test_notebooks_are_committed_without_outputs(notebook: Path) -> None:
    """Outputs make diffs unreadable and are a back door for git-ignored data."""
    import json

    cells = json.loads(notebook.read_text())["cells"]
    for index, cell in enumerate(cells):
        assert cell.get("outputs", []) == [], f"{notebook.name} cell {index} has outputs"
        assert cell.get("execution_count") is None, f"{notebook.name} cell {index} has a count"


def test_every_tier_module_exposes_run_all() -> None:
    """The uniform entry point the notebooks and ``report.py`` rely on."""
    from startup_outcomes import audit, descriptive, diagnostic, report
    from startup_outcomes.models import prescriptive, supervised, unsupervised

    for module in (audit, descriptive, diagnostic, supervised, unsupervised, prescriptive, report):
        assert callable(module.run_all), module.__name__


def test_the_package_exports_every_module() -> None:
    import startup_outcomes

    assert "descriptive" in startup_outcomes.__all__
    assert "report" in startup_outcomes.__all__
