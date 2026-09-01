from pathlib import Path
import unittest

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "09_phase_timing_evaluation.ipynb"


class PhaseTimingNotebookTests(unittest.TestCase):
    def test_notebook_is_source_only_and_code_cells_compile(self) -> None:
        notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)

        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            self.assertIsNone(cell.execution_count)
            self.assertEqual(cell.outputs, [])
            compile(cell.source, str(NOTEBOOK_PATH), "exec")


if __name__ == "__main__":
    unittest.main()
