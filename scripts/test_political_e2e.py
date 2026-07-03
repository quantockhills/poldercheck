"""One-off e2e test: political node only (manifestos off, CBS off) -> final answer."""

import asyncio
import json

from src.graph import run_query

QUERY = (
    "What has the Dutch parliament debated about the rise in mental health problems "
    "among young people, and what solutions have parties proposed?"
)


async def main() -> None:
    result = await run_query(
        QUERY,
        language="en",
        mode="deep",
        include_manifestos=False,
        include_tk=True,
        include_cbs=False,
        debug=True,
    )
    print("\n" + "=" * 80)
    print("POLITICAL RESPONSE")
    print("=" * 80)
    print(result["political_response"])
    print("\n" + "=" * 80)
    print("FINAL RESPONSE")
    print("=" * 80)
    print(result["final_response"])
    trace = result.get("political_trace") or {}
    if trace:
        print("\n" + "=" * 80)
        print("TRACE SUMMARY")
        print("=" * 80)
        print(json.dumps({k: v for k, v in trace.items() if not isinstance(v, list)}, indent=2, default=str))
        for k, v in trace.items():
            if isinstance(v, list):
                print(f"{k}: {len(v)} items")


if __name__ == "__main__":
    asyncio.run(main())
