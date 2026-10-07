#!/usr/bin/env python
"""Execute notebooks headlessly, saving executed copies to notebooks/_executed/.

Usage:
    uv run python scripts/run_notebooks.py                      # every notebooks/*.ipynb
    uv run python scripts/run_notebooks.py notebooks/09_a1_playground.ipynb ...

The source notebooks are never modified, so they stay output-free in git; the executed
copies (gitignored) are what you open to read the outputs. Each notebook runs in a fresh
kernel using this project's own Python (the .venv), with its working directory set to
notebooks/, as when opened in VS Code. A notebook stops at its first failing cell (later
cells usually depend on it); the error is kept in the executed copy and the run moves on
to the next notebook. A summary is printed and written to notebooks/_executed/SUMMARY.md.

This uses jupyter_client (already installed with ipykernel) rather than nbclient, which
is not a project dependency.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import nbformat
from jupyter_client.kernelspec import KernelSpecManager
from jupyter_client.manager import KernelManager

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "notebooks" / "_executed"


def _venv_kernel_manager(spec_root: Path) -> KernelManager:
    """A kernel that is guaranteed to run this interpreter, whatever kernelspecs exist."""
    spec_dir = spec_root / "gc2-venv"
    spec_dir.mkdir(parents=True, exist_ok=True)
    (spec_dir / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "GC2 .venv",
                "language": "python",
            }
        )
    )
    ksm = KernelSpecManager(kernel_dirs=[str(spec_root)])
    return KernelManager(kernel_name="gc2-venv", kernel_spec_manager=ksm)


def _run_cell(kc, source: str, timeout: float) -> tuple[list, int | None, str | None]:
    """Execute one cell; return (outputs, execution_count, error_or_None)."""
    msg_id = kc.execute(source)
    outputs: list = []
    error = None
    while True:
        try:
            msg = kc.get_iopub_msg(timeout=timeout)
        except queue.Empty:
            return outputs, None, f"timed out after {timeout:.0f}s waiting for output"
        if msg["parent_header"].get("msg_id") != msg_id:
            continue
        kind, content = msg["msg_type"], msg["content"]
        if kind == "status" and content["execution_state"] == "idle":
            break
        if kind == "clear_output":
            outputs = []
        elif kind in ("stream", "display_data", "execute_result", "error"):
            outputs.append(nbformat.v4.output_from_msg(msg))
            if kind == "error":
                error = f"{content['ename']}: {content['evalue']}"
    count = None
    while True:
        try:
            reply = kc.get_shell_msg(timeout=timeout)
        except queue.Empty:
            break
        if reply["parent_header"].get("msg_id") == msg_id:
            count = reply["content"].get("execution_count")
            break
    return outputs, count, error


def run_notebook(path: Path, timeout: float, spec_root: Path) -> dict:
    nb = nbformat.read(path, as_version=4)
    km = _venv_kernel_manager(spec_root)
    km.start_kernel(cwd=str(path.parent))
    kc = km.client()
    kc.start_channels()
    t0 = time.monotonic()
    result: dict[str, Any] = {
        "notebook": path.name,
        "status": "ok",
        "failed_cell": None,
        "error": None,
    }
    try:
        kc.wait_for_ready(timeout=120)
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        for n, cell in enumerate(code_cells, 1):
            print(f"  [{path.name}] cell {n}/{len(code_cells)}", flush=True)
            outputs, count, error = _run_cell(kc, cell.source, timeout)
            cell.outputs, cell.execution_count = outputs, count
            if error:
                result.update(status="error", failed_cell=n, error=error)
                break
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)
    result["seconds"] = round(time.monotonic() - t0, 1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, OUT_DIR / path.name)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("notebooks", nargs="*", type=Path)
    parser.add_argument(
        "--cell-timeout",
        type=float,
        default=3600.0,
        help="seconds to wait for a single cell's output (default 3600)",
    )
    args = parser.parse_args()
    paths = args.notebooks or sorted((REPO_ROOT / "notebooks").glob("[0-9]*.ipynb"))

    results = []
    with tempfile.TemporaryDirectory(prefix="gc2-kernelspec-") as spec_root:
        for path in paths:
            print(f"running {path.name} ...", flush=True)
            res = run_notebook(Path(path).resolve(), args.cell_timeout, Path(spec_root))
            print(
                f"  -> {res['status']} in {res['seconds']}s"
                + (f" (cell {res['failed_cell']}: {res['error']})" if res["error"] else "")
            )
            results.append(res)

    lines = [
        "# Notebook run summary",
        "",
        f"Run at {time.strftime('%Y-%m-%d %H:%M')}.",
        "",
        "| notebook | status | seconds | failed cell | error |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        err = (r["error"] or "").replace("|", "\\|")[:160]
        lines.append(
            f"| {r['notebook']} | {r['status']} | {r['seconds']} | "
            f"{r['failed_cell'] or ''} | {err} |"
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if all(r["status"] == "ok" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
