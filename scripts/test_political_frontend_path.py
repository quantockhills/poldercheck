"""Reproduce the Streamlit frontend's exact invocation path for the political node.

Mimics src/app.py _search_thread: worker thread + asyncio.run + on_status +
_CancelCallback/_StatusCallback, with the same settings as the failed runs in
data/history/ (en, deep, pedagogical, manifestos off, TK on, CBS off).
"""

import asyncio
import threading

from langchain_core.callbacks import BaseCallbackHandler

from src.graph import run_query

QUERY = (
    "What has the Dutch parliament debated about the rise in mental health problems "
    "among young people, and what solutions have parties proposed?"
)


class _CancelCallback(BaseCallbackHandler):
    def __init__(self, stop_event: threading.Event):
        super().__init__()
        self._stop = stop_event

    def on_chain_start(self, serialized, inputs, **kwargs):
        if self._stop.is_set():
            raise StopIteration("Search cancelled by user.")


class _StatusCallback(BaseCallbackHandler):
    def __init__(self, write_fn):
        super().__init__()
        self._write = write_fn

    def on_chain_start(self, serialized, inputs, **kwargs):
        name = (serialized or {}).get("name", "")
        if name:
            self._write(f"[node] {name}")


def main() -> None:
    stop_event = threading.Event()
    msgs: list = []
    out: dict = {"done": False}

    def _go():
        try:
            out["result"] = asyncio.run(
                run_query(
                    QUERY,
                    language="en",
                    mode="deep",
                    pedagogical=True,
                    include_manifestos=False,
                    include_tk=True,
                    include_cbs=False,
                    cbs_mode="duckdb",
                    num_datasets=5,
                    on_status=msgs.append,
                    extra_callbacks=[_CancelCallback(stop_event), _StatusCallback(msgs.append)],
                    debug=False,
                )
            )
        except Exception as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            out["done"] = True

    t = threading.Thread(target=_go, daemon=True)
    t.start()
    t.join()

    print("\n=== STATUS MSGS ===")
    for m in msgs:
        print(" ", m)
    if out.get("error"):
        print("\n=== THREAD ERROR ===\n", out["error"])
    result = out.get("result") or {}
    political = result.get("political_response", "")
    print("\n=== POLITICAL RESPONSE (len {}) ===".format(len(political)))
    print(political[:1500])
    print("\n=== PASSAGES:", len(result.get("political_passages") or []), "| TRACE keys:",
          list((result.get("political_trace") or {}).keys()))


if __name__ == "__main__":
    main()
