"""Wait for owned asynchronous work without interrupting its cleanup twice."""

from __future__ import annotations

import asyncio


def cancel_once(task: asyncio.Task[object]) -> None:
    if not task.done() and not task.cancelling():
        task.cancel()


async def finish(task: asyncio.Future[object]) -> None:
    while not task.done():
        try:
            # Wait without adding this frame and its callers to a failed task's traceback.
            await asyncio.wait((task,))
        except asyncio.CancelledError:
            continue
        except BaseException:
            break
    if not task.cancelled():
        task.exception()
