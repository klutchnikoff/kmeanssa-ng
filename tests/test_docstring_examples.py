"""The head-of-module examples must run: they are what a new user copies first."""

import re
import textwrap

import kmeanssa_ng


def _python_blocks(doc: str) -> list[str]:
    return [
        textwrap.dedent(block)
        for block in re.findall(r"```python\n(.*?)```", doc, re.S)
    ]


def test_package_docstring_example_runs():
    blocks = _python_blocks(kmeanssa_ng.__doc__)
    assert blocks, "the package docstring must contain a python example"
    exec(blocks[0], {})


def test_simulated_annealing_docstring_example_runs():
    blocks = _python_blocks(kmeanssa_ng.SimulatedAnnealing.__doc__)
    assert blocks, "the SimulatedAnnealing docstring must contain a python example"
    exec(blocks[0], {})
