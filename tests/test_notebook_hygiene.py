import json
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIRECTORY = PROJECT_ROOT / "notebooks"


class NotebookHygieneTests(unittest.TestCase):
    def test_tracked_notebooks_are_source_only(self) -> None:
        notebook_paths = sorted(NOTEBOOK_DIRECTORY.glob("*.ipynb"))
        self.assertTrue(notebook_paths, "no notebooks found")

        problems = []
        for notebook_path in notebook_paths:
            raw_notebook = notebook_path.read_text(encoding="utf-8")
            notebook = json.loads(raw_notebook)

            if "/Users/" in raw_notebook:
                problems.append(f"{notebook_path.name}: contains an absolute user path")

            for cell_index, cell in enumerate(notebook.get("cells", [])):
                if cell.get("cell_type") != "code":
                    continue
                if cell.get("outputs"):
                    problems.append(f"{notebook_path.name}: cell {cell_index} has outputs")
                if cell.get("execution_count") is not None:
                    problems.append(
                        f"{notebook_path.name}: cell {cell_index} has an execution count"
                    )

        self.assertEqual(problems, [], "\n".join(problems))


if __name__ == "__main__":
    unittest.main()
