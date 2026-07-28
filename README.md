# gpuq — Lightweight GPU Job Scheduler

A single-node GPU job queue for multi-user Linux servers. Users submit shell
commands that wait their turn, then run automatically when a GPU is free —
no containers, no Kubernetes, no job-specific daemons. Jobs run as the
submitting user; the daemon is a pure traffic controller that never touches
user processes.

**What it is:** a thin coordination layer on top of whatever you already run
(PyTorch scripts, shell pipelines, anything). Submit a command; it blocks in
the background until a GPU is available; `CUDA_VISIBLE_DEVICES` is set for
you; the log is saved; the GPU is released when the process exits.

**What it is not:** a cluster scheduler, a container runtime, a job
prioritisation engine, or a hard memory enforcer.

---

## How It Works

```
User shell                Background waiter            Daemon (root)
    |                           |                           |
    | gpuq submit "cmd"         |                           |
    |-------------------------->| fork, parent returns       |
    |  (shell free immediately) |                           |
    |                           |-- register(job) --------->|
    |                           |   (blocks, waiting)       | (queues job)
    |                           |                           |
    |                           |                           | GPU becomes free
    |                           |                           | fairness check
    |                           |<-- go(CUDA_VISIBLE_DEVICES=N) ---|
    |                           |                           |
    |                           | redirect stdout/stderr    |
    |                           | to log file               |
    |                           | exec(cmd) as self         |
    |                           | ...job runs...            |
    |                           |-- done(job_id, exit_code)->|
    |                           | exit                      | (GPU freed)
```

**Key property:** The daemon never spawns or owns user processes. The
background waiter is forked from the user's own shell, so it already runs as
the correct user before it ever contacts the daemon. The daemon only tells
already-running waiter processes when and where to execute.

`gpuqd` runs as root via systemd, owns the Unix socket and SQLite database,
polls `nvidia-smi`, and applies fairness scheduling. `gpuq` is the user CLI
that submits jobs, queries status, and tails logs.

---

## Installation

### Production (system-wide, requires root)

```bash
# 1. Install package, create system directories, install systemd unit
sudo ./install.sh

# 2. Start the daemon
sudo systemctl enable --now gpuq

# 3. Verify
sudo systemctl status gpuq
gpuq status
```

Every local user can submit jobs immediately once the service is running —
there is no group to join and no per-user setup.

The installer:
- Creates `/var/lib/gpuq` (DB) and `/var/log/gpuq` (daemon log), owned
  `root:root` (only the daemon touches them)
- Installs the package into a dedicated system virtualenv at `/opt/gpuq/venv`
  and symlinks the `gpuq` / `gpuqd` console scripts into `/usr/local/bin`, so
  every user runs them globally with **no venv activation**. The venv also
  sidesteps PEP 668 (`externally-managed-environment`), which blocks a plain
  system-wide `pip install` on modern Debian/Ubuntu/Fedora. (On Debian/Ubuntu
  you may first need `sudo apt install python3-venv`.)
- Installs the systemd unit at `/etc/systemd/system/gpuq.service`, with
  `ExecStart` pointing at `/opt/gpuq/venv/bin/gpuqd`
- Installs `/etc/gpuq/gpuq.conf` from the bundled example (only if the file
  does not already exist)

End users do **not** need a virtualenv: `gpuq` is on the system `PATH` and the
runtime is stdlib-only. To upgrade, re-run `sudo ./install.sh`; to remove,
delete `/opt/gpuq`, the two `/usr/local/bin` symlinks, and the systemd unit.

The socket at `/run/gpuq/gpuq.sock` is created world-accessible (mode `0666`)
inside a world-traversable runtime directory, so **any local user** can connect
without belonging to any group. This suits a trusted single-node box. To
restrict access instead, set a tighter `socket_mode` in `gpuq.conf` and manage
socket ownership yourself.

### Quick local / dev run (no root needed)

Override all paths with environment variables so both the daemon and client
run entirely in user space:

```bash
# Terminal 1: start the daemon
export GPUQ_SOCKET=/tmp/gpuq-dev.sock
export GPUQ_DB=/tmp/gpuq-dev.db
export GPUQ_NVIDIA_SMI=tests/fakebin/nvidia-smi   # or a real one
gpuqd --socket "$GPUQ_SOCKET" --db "$GPUQ_DB"

# Terminal 2: submit a job (same env vars apply to the client automatically)
export GPUQ_SOCKET=/tmp/gpuq-dev.sock
gpuq submit "python train.py --epochs 10" --name dev_run --gpus 1
gpuq status
gpuq logs <id> -f
```

Both `gpuq` and `gpuqd` also accept `--socket` and `--config` flags, and
`gpuqd` additionally accepts `--db`. The env var `GPUQ_CONF` can point to an
alternate config file.

---

## Usage

All subcommands support `--socket <path>` to override the socket location.

### `submit` — queue a job

```
gpuq submit <command> [--name NAME] [--gpus N] [--mem SIZE] [--share]
```

The shell returns immediately after the job is accepted. A background waiter
process (detached from your terminal via `setsid`) holds the connection and
will exec your command when the daemon grants a slot.

Validation errors (bad `--mem`, `--share` without `--mem`, requesting more
GPUs than are managed, `--mem` larger than any GPU, daemon unreachable) are
reported in your shell **before** the fork and exit non-zero.

```bash
# Single GPU, no memory hint
gpuq submit "python train.py --lr 0.001" --name baseline
# Job 7a3f queued (position 1)
# Logs: /home/alice/.gpuq/logs/baseline_20260516_143022.log

# Two GPUs, required memory specified
gpuq submit "python ddp_train.py" --name ddp_run --gpus 2 --mem 40G

# Shared GPU (co-locate with other --share jobs if memory fits)
gpuq submit "python eval.py" --name eval --gpus 1 --mem 8G --share
```

**Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--name NAME` | (derived from cmd) | Label for the job and its log file |
| `--gpus N` | `1` | Number of GPUs required (multi-GPU is atomic: all N or none) |
| `--mem SIZE` | none | Per-GPU memory hint/requirement, e.g. `8G`, `512M`, `1.5G`. Unit suffix required. For exclusive jobs, steers best-fit only. For `--share` jobs, required and enforced by placement logic. |
| `--share` | off | Allow co-location with other `--share` jobs. Requires `--mem`. |

#### Using your own virtualenv / conda environment

`gpuq` is installed system-wide and runs on its own Python, completely separate
from your job's interpreter — so your jobs use whatever environment **you** have
active, not gpuq's. The waiter snapshots your shell's full environment (your
`PATH`, `VIRTUAL_ENV`, `CONDA_PREFIX`, `PYTHONPATH`, etc.) at submit time and
runs the command in a shell with it.

The usual pattern just works — activate first, then submit:

```bash
source ~/myproject/.venv/bin/activate     # or: conda activate myproject
gpuq submit "python train.py --epochs 50" --name baseline
```

Note the environment is captured **when you submit**, not when the job later
starts. If you'd rather not depend on the ambient environment (e.g. you submit
many jobs from different envs), make activation explicit in the command, or call
the interpreter by absolute path:

```bash
# Activate inside the job command (runs in a shell, so '&&' works)
gpuq submit "source ~/myproject/.venv/bin/activate && python train.py" --name baseline

# Or point straight at the venv/conda python
gpuq submit "~/myproject/.venv/bin/python train.py" --name baseline
```

### `status` — show the active queue

```bash
gpuq status           # all users
gpuq status --mine    # your jobs only, with a summary line
```

```
ID      USER     STATUS    POS  GPU  SUBMITTED         STARTED           NAME/CMD
7a3f    alice    queued    1    -    2026-05-16 14:30  -                 baseline
3b1c    bob      running   -    0    2026-05-16 14:15  2026-05-16 14:20  ./finetune.sh
9d2a    carol    running   -    1    2026-05-16 14:22  2026-05-16 14:25  python eval.py
```

### `mine` — your jobs with summary

Shorthand for `gpuq status --mine`, always includes the summary line.

```bash
gpuq mine
# Your jobs: 1 running, 1 queued
#
# ID      STATUS    POS  GPU  SUBMITTED         STARTED           NAME/CMD
# 7a3f    queued    1    -    2026-05-16 14:30  -                 baseline
# 3b1c    running   -    0    2026-05-16 14:15  2026-05-16 14:20  finetune
```

### `logs` — view a job's output

```bash
gpuq logs 7a3f          # print log file so far
gpuq logs 7a3f -f       # follow (like tail -f), Ctrl-C to stop
```

Job IDs can be given as a short prefix (the first 4 characters shown in the
output). The log file also works for finished jobs retrieved from history.

Log files are stored in `~/.gpuq/logs/` and include a metadata header
(job id, command, user, assigned GPUs) followed by merged stdout + stderr of
the job.

### `cancel` — remove a queued job, or kill a running one

Works on queued and running jobs. Users can only cancel their own jobs; root
can cancel any job.

```bash
gpuq cancel 7a3f    # cancel job with id starting with 7a3f
gpuq cancel --all   # cancel all your queued jobs
# Cancelled 2 queued job(s).
```

Cancelling a **running** job asks first:

```bash
gpuq cancel 7a3f
# Job 7a3f (trainjob) is running on GPU 0:
#   python train.py
# Kill it? [y/N] y
# Killed job 7a3f (trainjob)
```

The job's process group gets `SIGTERM`, then `SIGKILL` 10 seconds later if it
is still alive — so a script that traps `SIGTERM` has a chance to flush
checkpoints. The whole process group is signalled, so child processes spawned
by your command are cleaned up too. The job lands in `history` as `cancelled`.

Use `--force` (`-f`) to skip the confirmation in scripts. Without a terminal
and without `--force`, `cancel` refuses rather than hanging on a prompt nobody
can answer.

```bash
gpuq cancel 7a3f --force
```

`--all` only ever touches **queued** jobs; it will not kill anything running.

### `history` — recently finished jobs

```bash
gpuq history                    # last 20 finished/failed/cancelled jobs (all users)
gpuq history --mine             # your jobs only
gpuq history --limit 5 --mine  # your 5 most recent
```

```
ID      STATUS    GPU  SUBMITTED         STARTED           FINISHED          EXIT  NAME/CMD
2a1b    done      1    2026-05-16 12:00  2026-05-16 12:05  2026-05-16 13:42  0     long_run
8c9e    failed    0    2026-05-16 11:00  2026-05-16 11:02  2026-05-16 11:04  1     python_eval_py
```

---

## Scheduling Explained

### Fairness

When a GPU slot opens up, the daemon ranks all waiting jobs by
`(number of that user's currently running jobs, submission time)`. The user
with the fewest running jobs goes first; ties are broken FIFO. This prevents
any single user from monopolising all GPUs.

After one job is placed, the counts are updated and the ordering is
re-evaluated before placing the next, so fairness applies across a whole
batch of placements in a single scheduler pass.

### Best-Fit Placement

Among GPUs eligible for a job, the scheduler picks the smallest GPU(s) that
fit. For exclusive jobs "smallest" means the GPU with the least total memory
(keeping large cards free for jobs that need them). For shared jobs it means
the GPU with the least remaining capacity after the new reservation (tight
packing, so larger GPUs stay available for exclusive or large jobs).

Multi-GPU jobs are **atomic**: the job gets all N GPUs simultaneously or is
not placed at all.

### Skip, Don't Block

If the fairest job cannot be placed right now (e.g. it needs 2 GPUs but only
1 is free), the scheduler skips it and tries the next candidate. There is no
head-of-line blocking. Trade-off: a large or exclusive job can be starved by
a continuous stream of smaller jobs. No aging or priority escalation is
implemented in v1.

### Exclusive vs. Shared Jobs

**Exclusive (default):** The job owns each assigned GPU entirely. A GPU is
eligible for an exclusive job only if it has no gpuq reservations and its
live memory usage is below `idle_memory_threshold_mb` (default 500 MB —
this catches orphaned external processes). `--mem` is optional for exclusive
jobs; if given, it steers best-fit by excluding GPUs too small to hold it,
but does not limit the job's actual VRAM usage.

**Shared (`--share`, requires `--mem`):** The job may co-locate on a GPU
only with other `--share` jobs. Placement passes a **dual check**: both
(a) the sum of gpuq `--mem` reservations must leave room, and (b) the live
`nvidia-smi` free memory must be sufficient (guards against OOM from external
processes or not-yet-accounted memory).

**Advisory limits:** gpuq advises placement but does **not** hard-enforce
VRAM limits at runtime. A shared job that exceeds its declared `--mem`, or an
external process using the same card, can still cause an OOM that kills a
co-located neighbour. If tight VRAM isolation is required, use exclusive mode.

---

## Configuration

Config file: `/etc/gpuq/gpuq.conf` (INI format). All keys are optional;
built-in defaults are shown.

```ini
[daemon]
socket_path = /run/gpuq/gpuq.sock
db_path = /var/lib/gpuq/queue.db
log_path = /var/log/gpuq/daemon.log
poll_interval_seconds = 5
# socket_mode = 0666    ; default: open to all local users (no group needed).
                        ; set tighter (e.g. 0660) + manage ownership to restrict.

[gpus]
# managed_gpus = 0,1,2,3   ; default: all GPUs detected by nvidia-smi
idle_memory_threshold_mb = 500

[fairness]
max_gpus_per_user = 0    ; 0 = unlimited
```

| Key | Section | Default | Description |
|-----|---------|---------|-------------|
| `socket_path` | `[daemon]` | `/run/gpuq/gpuq.sock` | Unix domain socket path |
| `db_path` | `[daemon]` | `/var/lib/gpuq/queue.db` | SQLite database path |
| `log_path` | `[daemon]` | `/var/log/gpuq/daemon.log` | Daemon log file |
| `poll_interval_seconds` | `[daemon]` | `5` | How often the scheduler wakes to check GPU state even without new events |
| `socket_mode` | `[daemon]` | `0666` | Octal permission mode for the socket; default is open to all local users |
| `managed_gpus` | `[gpus]` | (all) | Comma-separated list of GPU indices to manage; omit to use all GPUs reported by nvidia-smi |
| `idle_memory_threshold_mb` | `[gpus]` | `500` | A GPU whose live `nvidia-smi` memory use exceeds this is treated as occupied (not free for exclusive jobs) |
| `max_gpus_per_user` | `[fairness]` | `0` | Maximum GPUs a single user may hold simultaneously; `0` means unlimited |

**Override precedence (highest first):** CLI flag > environment variable
(`GPUQ_SOCKET`, `GPUQ_DB`, `GPUQ_LOG`, `GPUQ_CONF`) > config file > built-in default.

---

## Caveats & Troubleshooting

### Memory limits are advisory, not enforced

gpuq uses `--mem` declarations to decide placement. It does not set cgroups
or any hardware memory limit. A job that exceeds its declared memory, or an
external process on the same GPU, can still cause an OOM that kills other
co-located (`--share`) jobs. For strong isolation, use exclusive mode (the
default).

### Daemon restart marks in-flight jobs failed

If `gpuqd` is restarted (crash, `systemctl restart`) while jobs are running,
those jobs are marked `failed` in the database on startup because there is no
reconnect protocol between the daemon and the waiter processes. The underlying
processes may keep running until they finish on their own (they will still
produce output to their log files), but gpuq will no longer track them or
free their GPU slots automatically.

### Common issues

**`gpuq: error: cannot reach gpuqd at /run/gpuq/gpuq.sock`**
The daemon is not running. Check: `sudo systemctl status gpuq`. Start with
`sudo systemctl start gpuq`.

**`Permission denied` connecting to the socket**
By default the socket is world-accessible, so this should not happen. If it
does, an admin has set a restrictive `socket_mode` in `gpuq.conf` — check
`stat -c '%a' /run/gpuq/gpuq.sock` (expect `666`) and the `socket_mode`
setting.

**`nvidia-smi binary not found`**
`nvidia-smi` is not on `PATH`. Install the NVIDIA driver utilities, or set
`GPUQ_NVIDIA_SMI=/full/path/to/nvidia-smi`.

**Job stuck in `queued` forever**
Possible causes: all GPUs are occupied by external processes (above
`idle_memory_threshold_mb`); a multi-GPU job is waiting for enough GPUs to be
free simultaneously; `max_gpus_per_user` is set and you've reached the limit.
Run `gpuq status` and check GPU occupancy with `nvidia-smi`.

**Log file is empty or missing**
The job may not have started yet (still `queued`). If it shows `running`, the
log is at the path printed by `submit`. If it shows `failed` immediately after
submission, check whether the daemon was restarted while it was running.

---

## Development

### Running Tests

```bash
# Install with dev dependencies (use a virtualenv to avoid PEP 668 issues)
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]

# Unit tests (pure logic, no GPU, no root needed)
pytest tests/unit

# Integration tests (Linux only; use a fake nvidia-smi shim, no real GPU)
pytest tests/integration

# Both
pytest tests/
```

The fake `nvidia-smi` shim lives at `tests/fakebin/nvidia-smi` and is
selected by setting `GPUQ_NVIDIA_SMI=tests/fakebin/nvidia-smi`.

If you don't want an editable install, you can also run tests with:

```bash
PYTHONPATH=src pytest tests/unit
```

### Stdlib-Only Constraint

The runtime has **no pip dependencies** — only the Python standard library
(`sqlite3`, `socket`, `subprocess`, `threading`, `json`, `os`, etc.).
`pytest` is the only dev dependency. Do not add runtime dependencies.
