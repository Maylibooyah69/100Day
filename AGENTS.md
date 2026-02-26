# AGENTS

## Cursor Cloud specific instructions

This is a Python starter repository ("100Day" project) containing `blender_test.py`. The repo currently has no dependencies, no build system, and no test framework.

- **Language**: Python 3.12+ (system Python at `/usr/bin/python3`)
- **Run scripts**: `python3 <script>.py` (no build step required)
- **No dependency file exists yet**; if `requirements.txt` or `pyproject.toml` is added, install with `pip install -r requirements.txt` or `pip install -e .`
- **No lint/test tooling configured**; if added, follow whichever framework is introduced (e.g. `pytest`, `flake8`, `ruff`)
- The file `blender_test.py` suggests Blender scripting; Blender is a heavy GUI application and is not installed in this cloud environment. Python scripts that import `bpy` (Blender's Python module) cannot run standalone without Blender.
