"""Run the standalone notebook in a fresh, local IPython kernel without starting a server."""

import argparse
import asyncio
import os
import sys
from pathlib import Path


def execute(notebook, source, input_dir, output, offline):
    import nbformat
    from ipykernel.inprocess.manager import InProcessKernelManager

    # A new invocation is a clean process. The kernel uses the active Python and
    # exchanges Jupyter messages in memory, including the PNG and HTML displays.
    os.chdir(source.parent)
    os.environ["GHIR_OUTPUT"] = str(output)
    if input_dir is not None:
        os.environ["GHIR_INPUTS"] = str(input_dir)
    if offline:
        os.environ["GHIR_OFFLINE"] = "1"
    manager = InProcessKernelManager()
    manager.start_kernel()
    client = manager.client()
    client.start_channels()
    try:
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            heading = notebook.cells[index - 1].source.splitlines()[0].lstrip("# ")
            print(heading, flush=True)
            message_id = client.execute(cell.source, store_history=True, allow_stdin=False)
            reply = client.get_shell_msg(timeout=30)
            if reply["parent_header"].get("msg_id") != message_id:
                raise RuntimeError(f"Unexpected kernel reply at {heading}")
            cell.execution_count = reply["content"]["execution_count"]
            while True:
                message = client.get_iopub_msg(timeout=60)
                kind = message["msg_type"]
                if kind == "status" and message["content"]["execution_state"] == "idle":
                    break
                if kind in ("stream", "display_data", "execute_result", "error"):
                    cell.outputs.append(nbformat.v4.output_from_msg(message))
            if reply["content"]["status"] != "ok":
                error = reply["content"]
                raise RuntimeError(
                    f"{heading}: {error.get('ename', 'Kernel error')}: {error.get('evalue', '')}"
                )
        info = manager.kernel.language_info
        notebook.metadata["language_info"] = info
    finally:
        client.stop_channels()

        # Let pending output-thread tasks finish cancellation before the loop
        # closes. This also keeps an interrupted notebook free of teardown noise.
        async def drain_background_tasks():
            tasks = asyncio.all_tasks() - {asyncio.current_task()}
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        io_loop = manager.kernel.iopub_thread.io_loop.asyncio_loop
        try:
            asyncio.run_coroutine_threadsafe(drain_background_tasks(), io_loop).result(timeout=10)
        finally:
            manager.shutdown_kernel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook", type=Path, default=Path(__file__).with_name("GHIR_Public_Notebook.ipynb")
    )
    parser.add_argument("--inputs", type=Path, help="Existing or new GEO cache")
    parser.add_argument("--output", type=Path, help="Results directory for this run")
    parser.add_argument("--offline", action="store_true", help="Require checksummed local data")
    args = parser.parse_args()
    try:
        import nbformat
        import ipykernel
    except ImportError as exc:
        parser.exit(1, f"{exc}. Install packages with: python -m pip install -r requirements.txt\n")
    source = args.notebook.expanduser().resolve()
    if not source.is_file():
        parser.error(f"Notebook not found: {source}")
    output = (
        Path(args.output or os.environ.get("GHIR_OUTPUT") or source.parent / "GHIR_reproduction")
        .expanduser()
        .resolve()
    )
    inputs = args.inputs.expanduser().resolve() if args.inputs is not None else None
    output.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(source, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    try:
        execute(notebook, source, inputs, output, args.offline)
    except (Exception, KeyboardInterrupt):
        partial = output / "GHIR_interrupted.ipynb"
        nbformat.write(notebook, partial)
        print(f"Interrupted run saved to {partial}", file=sys.stderr)
        raise
    nbformat.validate(notebook)
    dest = output / "GHIR_executed.ipynb"
    nbformat.write(notebook, dest)
    print(f"Executed notebook: {dest}")


if __name__ == "__main__":
    main()
