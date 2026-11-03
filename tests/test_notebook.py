import json
from pathlib import Path


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "lopes2025_reproduction_standalone.ipynb"


def test_standalone_notebook_is_documented_and_self_contained():
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook["metadata"]["language_info"]["version"].startswith("3.14")
    assert sum(cell["cell_type"] == "markdown" for cell in notebook["cells"]) >= 6
    source = "\n".join(
        line
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        for line in cell["source"]
    )
    assert "gastric_repro" not in source
    assert "subprocess" not in source
    assert "figure_3c_saliva_heatmap.png" in source
    assert "figure_3d_saliva_volcano.png" in source
    assert "differential_expression" in source
