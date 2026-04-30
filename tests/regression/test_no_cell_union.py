"""Regression: no ua.cell.Cell references remain in the codebase."""
import subprocess

def test_no_cell_union_in_source():
    result = subprocess.run(
        ["grep", "-r", "ua.cell.Cell", "src/unialg/"],
        capture_output=True, text=True,
    )
    assert result.stdout == "", f"ua.cell.Cell found:\n{result.stdout}"


def test_no_para_imports_in_source():
    result = subprocess.run(
        ["grep", "-rn", "--include=*.py", r"from.*_para\b", "src/unialg/"],
        capture_output=True, text=True,
    )
    # Filter false positives: "rebind_params" contains "_para"
    real_hits = [
        line for line in result.stdout.splitlines()
        if "_para_" in line or "._para " in line or "._para." in line
    ]
    assert real_hits == [], f"_para imports found:\n" + "\n".join(real_hits)
