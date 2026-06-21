from __future__ import annotations

import os
import dask
from dask.distributed import Client, LocalCluster, default_client


def shutdown_default_client() -> None:

    try:
        default_client().shutdown()
    except Exception:
        pass


def make_local_dask_client(
    n_workers: int | None = None,
    threads_per_worker: int = 1,
    processes: bool = True,
    memory_limit: str | int | None = None,
    dashboard_address: str | None = ":8787",
    local_directory: str | None = "dask-tmp",
    *,
    memory_target: float = 0.75,
    memory_spill: float = 0.85,
    memory_pause: float = 0.92,
    memory_terminate: float = 0.98,
) -> Client:
    if n_workers is None:
        n_workers = max(2, (os.cpu_count() or 2) // 2)

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        processes=processes,
        memory_limit=memory_limit,
        dashboard_address=dashboard_address,
        local_directory=local_directory,
    )

    dask.config.set(
        {
            "distributed.worker.memory.target": memory_target,
            "distributed.worker.memory.spill": memory_spill,
            "distributed.worker.memory.pause": memory_pause,
            "distributed.worker.memory.terminate": memory_terminate,
        }
    )

    return Client(cluster)