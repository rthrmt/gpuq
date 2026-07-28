"""gpuq — a lightweight single-node GPU job scheduler.

The package exposes two console entrypoints:

* ``gpuq``  — the user-facing CLI and the per-job *waiter* process (``gpuq.cli``).
* ``gpuqd`` — the root scheduler daemon (``gpuq.daemon``).

The runtime is intentionally stdlib-only.
"""

__version__ = "0.1.0"
